"""Implement the genetic algorithm described in

Open Sesame! Universal Black Box Jailbreaking of Large Language Models
Raz Lapid, Ron Langberg, and Moshe Sipper

As far as I can tell, this is a reasonably standard genetic algorithm.
More details on genetic algorithms can be found in the good-old matlab documentation here:
https://www.mathworks.com/help/gads/genetic-algorithm-options.html#f6593
"""


import torch
import random

from .generic_optimizer import _GenericOptimizer

_default_setup = dict(device=torch.device("cpu"), dtype=torch.float32)


class LLSOptimizer(_GenericOptimizer):
    def __init__(
        self,
        *args,
        setup=_default_setup,
        save_checkpoint=False,
        population_size=500,
        steps=500,  # these are "generations" in GA language
        tournament_size=2,
        elitism=0.2,
        filter_cand=True,
        **kwargs,
    ):
        super().__init__(setup=setup, save_checkpoint=save_checkpoint)
        self.population_size = population_size
        self.steps = steps
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.filter_cand = True

    @torch.no_grad()
    def initialize_population(self, constraint):
        population = []
        for p in range(self.population_size):
            population.append(constraint.draw_random_sequence(device=self.setup["device"]))
        return torch.cat(population)

    @torch.no_grad()
    def get_filtered_cands(self, candidate_ids, constraint):
        if self.filter_cand:
            candidate_is_valid = constraint.is_tokenization_safe(candidate_ids)
            if sum(candidate_is_valid) > 0:
                return candidate_ids[candidate_is_valid], True
            else:
                print(f"No valid candidate accepted out of {len(candidate_ids)} candidates.")
                return candidate_ids, False
        else:
            return candidates, True

    @torch.no_grad()
    def sample_mutations(self, input_ids, constraint):
        weights = torch.as_tensor(
            [len(constraint.set_at_idx(idx)) for idx in range(constraint.num_tokens)], device=input_ids.device, dtype=torch.float
        )
        new_token_pos = torch.multinomial(weights, len(input_ids), replacement=True)
        new_token_val = torch.stack([random.choice(constraint.set_at_idx(location)) for location in new_token_pos])
        new_input_ids = input_ids.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val.unsqueeze(-1))
        return new_input_ids

    def solve(self, sigil, dryrun=False, **kwargs):
        """"""

        # Initialize pop
        population = self.initialize_population(sigil.constraint)

        best_loss = float("inf")
        prev_loss = float("inf")
        best_prompt_ids = sigil.constraint.draw_random_sequence(device=self.setup["device"])

        for generation in range(self.steps):
            # Evaluate fitness of each individual (like Algorithm 3);
            loss = torch.zeros(len(population), **self.setup)
            with torch.no_grad():
                for j, individual_ids in enumerate(population):
                    loss[j] = sigil.objective(input_ids=individual_ids[None]).mean()

            # Save best individual for later
            min_loss, min_loss_val = torch.min(loss, dim=0)
            if min_loss < best_loss:
                best_loss = min_loss.clone().detach()
                best_prompt_ids = population[min_loss_val][None]

            # Save elitist individuals;
            lambda_elites = int(self.population_size * self.elitism)
            elite_population = population[loss.argsort(descending=False, stable=False)[:lambda_elites]]

            # Select parents for reproduction via N-way tournaments:
            rand_permutation = torch.randperm(len(population), device=self.setup["device"])
            usable_indices = len(population) // self.tournament_size * self.tournament_size
            contestant_indices = rand_permutation[:usable_indices].view(self.tournament_size, -1)
            parent_indices = torch.gather(contestant_indices, 0, loss[contestant_indices].argmin(dim=0, keepdim=True))[0]
            parents = population[parent_indices]

            # Perform crossover between parents to derive offspring:
            # One-point crossover, as described in the paper
            crossover_points = torch.randint(0, len(sigil), (len(parents) // 2,))
            offspring = []
            for pair, crossover in zip(range(len(parents) // 2), crossover_points):
                parent_1 = parents[2 * pair]
                parent_2 = parents[2 * pair + 1]
                offspring.append(torch.cat([parent_1[:crossover], parent_2[crossover:]]))
                offspring.append(torch.cat([parent_1[crossover:], parent_2[:crossover]]))
            offspring = torch.stack(offspring)

            # Perform mutation on parents (could also have done this on crossover offspring, unclear from the paper);
            mutations = self.sample_mutations(parents, sigil.constraint)

            # Collect next generation
            population = torch.cat([elite_population, offspring, mutations])
            # pop_before_filtering = len(population)

            # Select valid (tokenization-safe) candidates
            population, valid_candidates_found = self.get_filtered_cands(population, sigil.constraint)
            # print(len(population), pop_before_filtering)

            # Cull to maximal population size:
            # Randomly cull non-elites:
            indices = torch.cat(
                [torch.arange(lambda_elites), torch.randint(lambda_elites, len(population), (self.population_size - lambda_elites,))]
            )
            population = population[indices]

            self.callback(sigil, population[0:1], best_prompt_ids, min_loss.detach(), generation, **kwargs)
            if dryrun:
                break

        return best_prompt_ids  # always return with leading dimension
