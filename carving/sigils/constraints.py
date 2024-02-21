"""Implement constraints for the optimization."""

import torch
import re
import random

from collections import defaultdict


# This is only a check for some metaspaces
whitespace_unicodes = ["▁", "Ġ"]


def get_constraint(constraint_name, tokenizer, embedding, num_tokens, blocklist=[]):
    if constraint_name in ["vocab", "none", None]:
        return VocabConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "ascii":
        return AsciiConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "chars":
        return CharConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "letters":
        return LetterConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "numbers":
        return NumberConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "chinese":
        return ChineseConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "simple-zalgo":
        return SimpleZalgoConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "anoikis":
        return SimpleZalgoConstraint(tokenizer, embedding, num_tokens, blocklist, skeleton="anoikis")
    elif constraint_name == "zalgo-skull":
        return ZalgoSkull(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "runes":
        return RunicConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "emoji":
        return EmojiConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "invisibles":
        return InvisibleConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "cuneiroform":
        return CuneiroformConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "hieroglyphs":
        return HieroglyphConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "chinese-full":
        return FullChineseConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "riley-invisible":
        return InvisibleTagConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "unwords":
        return UnwordConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "words":
        return WordConstraint(tokenizer, embedding, num_tokens, blocklist)
    elif constraint_name == "nonlatin":
        return NonLatinConstraint(tokenizer, embedding, num_tokens, blocklist)
    else:
        raise ValueError(f"Invalid name {constraint_name} given for a constraint set.")


def is_ascii(s):
    return s.isascii() and s.isprintable()


class _GenericConstraint(torch.nn.Module):
    """base class, just to outline the interface."""

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.num_tokens = num_tokens
        self.blocklist = blocklist

        self._cached_keys = False
        self.transposed_normalized_keys = torch.Tensor([])

    @property
    def set(self):
        """Which elements are in the set of admissible tokens?"""
        raise NotImplementedError()

    def set_at_idx(self, position_idx):
        """For all `num_tokens` positions, which elements are in the set of admissible tokens at this position?"""
        return self.set

    def select_topk(self, scores, k=5):
        """Return top-k admissible elements based on given scores."""
        return self.set[(scores[..., self.set]).topk(k, dim=-1).indices]

    def draw_random_sequence(self, device=torch.device("cpu")):
        """Draw a random sequence from this constraint. Useful to initialize optimization."""
        prompt_ids = self.set[torch.randint(len(self), (1, self.num_tokens)).to(device)]
        # Start with tokenizable prompt
        prompt_ids = self.project_onto_tokenizable_ids(prompt_ids)
        return prompt_ids

    def gather_random_element(self, indices, locations, scores=None):
        """Return random element from indices at location."""
        if scores is None:
            # weights = torch.ones(indices.shape[1])
            rand_choices = torch.randint(0, indices.shape[1], (len(locations), 1), device=indices.device)
        else:
            relevant_scores = torch.gather(scores, 1, indices)
            weights = (relevant_scores / relevant_scores.max(dim=-1, keepdim=True).values).exp()
            rand_choices = torch.gather(torch.multinomial(weights, len(locations), replacement=True), 0, locations[None]).T
        return torch.gather(indices[locations], 1, rand_choices).squeeze(dim=-1)

    @property
    def is_uniform(self):
        """Return true if the constraint set is equal at all positions, otherwise false."""
        return True

    def __len__(self) -> int:
        if not self.is_uniform:
            raise ValueError("No uniform length")
        else:
            return len(self.set)

    def _apply_blocklist(self, candidate_ids, ignore_cases=True, exact_matches=False):
        if len(self.blocklist) > 0:
            vocab_lookup = {val: key for key, val in self.tokenizer.vocab.items()}
            valid_ids, blocked_keys = [], []
            for idx in candidate_ids:
                key = vocab_lookup[idx]
                if exact_matches:
                    blocked = any(blocker == key for blocker in self.blocklist)
                elif ignore_cases:
                    blocked = any(blocker.lower() in key.lower() for blocker in self.blocklist)
                else:
                    blocked = any(blocker in key for blocker in self.blocklist)
                if not blocked:
                    valid_ids.append(idx)
                else:
                    blocked_keys.append(key)
            if len(blocked_keys) > 0:
                print(f"Explicitly blocked tokens {blocked_keys}")
            return valid_ids
        else:
            return candidate_ids

    @torch.no_grad()
    def normalize(self, embedded_inputs):
        return torch.nn.functional.normalize(embedded_inputs, p=2, dim=1)

    @torch.no_grad()
    # @torch.compile() # not necessarily faster?
    def normalized_project(self, embedded_inputs, topk=1):
        """if topk > 1, project onto one of the top-k entries randomly."""
        if not self._cached_keys:
            self.transposed_normalized_keys = self.normalize(self.embedding.weight[self.set]).T.contiguous()
            self._cached_keys = True
        bsz, seq_len, emb_dim = embedded_inputs.shape
        normalized_queries = self.normalize(embedded_inputs.view(-1, emb_dim))
        if topk == 1:
            indices = torch.matmul(normalized_queries, self.transposed_normalized_keys).argmax(dim=-1)
        else:
            all_indices = torch.matmul(normalized_queries, self.transposed_normalized_keys).topk(topk, dim=-1).indices
            selected_indices = torch.randint(0, topk, (bsz * seq_len,), device=embedded_inputs.device)
            indices = all_indices[torch.arange(bsz * seq_len, device=embedded_inputs.device), selected_indices]
        full_indices = self.set[indices.view(bsz, seq_len)]
        return self.embedding.weight[full_indices], full_indices

    def project_onto_tokenizable_ids(self, input_ids):
        """Input ids need to have a leading batch dimension."""
        retokenized_ids = self.tokenizer(
            self.tokenizer.batch_decode(input_ids, skip_special_tokens=True),
            add_special_tokens=False,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=input_ids.shape[1],
        )["input_ids"].to(input_ids.device)
        return retokenized_ids

    def is_tokenization_safe(self, input_ids):
        retokenized_ids = self.project_onto_tokenizable_ids(input_ids)
        input_ids_valid = (input_ids != retokenized_ids).sum(dim=1) == 0
        return input_ids_valid


class VocabConstraint(_GenericConstraint):
    """Generic constraint to the given vocabulary."""

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)
        self.vocab_size = tokenizer.vocab_size
        self.register_buffer("valid_ids", torch.as_tensor(self._apply_blocklist(range(0, self.vocab_size))), persistent=False)

    @property
    def set(self):
        return self.valid_ids


class AsciiConstraint(_GenericConstraint):
    """Restrict to ascii tokens. Constraint written to be compatible with the ascii constraint in `llm-attacks`."""

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[], use_legacy_decode_strategy=False):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)

        ascii_tokens = []
        if use_legacy_decode_strategy:
            for i in range(3, tokenizer.vocab_size):
                if is_ascii(tokenizer.decode([i])):
                    ascii_tokens.append(i)
            if tokenizer.bos_token_id is not None and tokenizer.bos_token_id not in ascii_tokens:
                ascii_tokens.append(tokenizer.bos_token_id)
            if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ascii_tokens:
                ascii_tokens.append(tokenizer.eos_token_id)
            if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in ascii_tokens:
                ascii_tokens.append(tokenizer.pad_token_id)
            if tokenizer.unk_token_id is not None and tokenizer.unk_token_id not in ascii_tokens:
                ascii_tokens.append(tokenizer.unk_token_id)
        else:
            vocab_lookup_without_whitespace = {
                val: key
                for key, val in tokenizer.vocab.items()
                if is_ascii(key.replace(whitespace_unicodes[0], "").replace(whitespace_unicodes[1], ""))
            }
            ascii_tokens = list(vocab_lookup_without_whitespace.keys())
            if tokenizer.eos_token_id in ascii_tokens:
                ascii_tokens.remove(tokenizer.eos_token_id)
        self.register_buffer("ascii_tokens", torch.as_tensor(sorted(self._apply_blocklist(ascii_tokens))), persistent=False)

    @property
    def set(self):
        return self.ascii_tokens


class ChineseConstraint(_GenericConstraint):
    """Restrict to chinese tokens (but note that most chinese characters are written with more than one token and not counted here)."""

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)

        def is_zh(str_in):
            return re.search("[\u4e00-\u9fff]", str_in)

        vocab_lookup_without_whitespace = {
            val: key
            for key, val in tokenizer.vocab.items()
            if is_zh(key.replace(whitespace_unicodes[0], "").replace(whitespace_unicodes[1], ""))
        }
        zh_tokens = list(vocab_lookup_without_whitespace.keys())
        self.register_buffer("zh_tokens", torch.as_tensor(sorted(self._apply_blocklist(zh_tokens))), persistent=False)

    @property
    def set(self):
        return self.zh_tokens


class CharConstraint(_GenericConstraint):
    """Constrain to tokens with len(str(token)) == 1"""

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[], include_whitespaces=False, only_ascii=True):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)
        if include_whitespaces:
            letter_tokens = {val: key for key, val in tokenizer.vocab.items() if len(key.replace(chr(288), "")) == 1}
        elif only_ascii:
            letter_tokens = {val: key for key, val in tokenizer.vocab.items() if len(key) == 1 and is_ascii(key)}
        else:
            letter_tokens = {val: key for key, val in tokenizer.vocab.items() if len(key) == 1}
        self.register_buffer("letter_tokens", torch.as_tensor(sorted(self._apply_blocklist(list(letter_tokens.keys())))), persistent=False)

    @property
    def set(self):
        return self.letter_tokens


class LetterConstraint(_GenericConstraint):
    """Constrain to only letters."""

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)
        allowed_ordinals = [*range(48, 58), *range(65, 91), *range(96, 123)]
        letter_tokens = {val: key for key, val in tokenizer.vocab.items() if len(key) == 1 and ord(key) in allowed_ordinals}
        self.register_buffer("letter_tokens", torch.as_tensor(sorted(self._apply_blocklist(list(letter_tokens.keys())))), persistent=False)

    @property
    def set(self):
        return self.letter_tokens


class NumberConstraint(_GenericConstraint):
    """Constrain to only numbers."""

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)
        allowed_nums = [str(num) for num in range(0, 9)]
        number_tokens = {
            val: key
            for key, val in tokenizer.vocab.items()
            if key.replace(whitespace_unicodes[0], "").replace(whitespace_unicodes[1], "") in allowed_nums
        }
        self.register_buffer("number_tokens", torch.as_tensor(sorted(self._apply_blocklist(list(number_tokens.keys())))), persistent=False)

    @property
    def set(self):
        return self.number_tokens


class NonLatinConstraint(_GenericConstraint):
    """Constrain to non-latin characters."""

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)
        all_letters = {chr(x) for x in [*range(48, 58), *range(65, 91), *range(96, 123)]}
        allowed_tokens = {val: key for key, val in tokenizer.vocab.items() if not any(kchr in all_letters for kchr in key)}
        self.register_buffer(
            "allowed_tokens", torch.as_tensor(sorted(self._apply_blocklist(list(allowed_tokens.keys())))), persistent=False
        )

    @property
    def set(self):
        return self.allowed_tokens


class UnwordConstraint(_GenericConstraint):
    """Constrain to tokens containing only non-alphabetic characters."""

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)
        allowed_tokens = {val: key for key, val in tokenizer.vocab.items() if not any(kchr.isalpha() for kchr in key)}
        self.register_buffer(
            "allowed_tokens", torch.as_tensor(sorted(self._apply_blocklist(list(allowed_tokens.keys())))), persistent=False
        )

    @property
    def set(self):
        return self.allowed_tokens


class WordConstraint(_GenericConstraint):
    """Constrain to tokens containing only alphabetic characters."""

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[], only_ascii=True):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)
        if only_ascii:
            allowed_tokens = {
                val: key
                for key, val in tokenizer.vocab.items()
                if all(
                    kchr.isalpha() and kchr.isascii()
                    for kchr in key.replace(whitespace_unicodes[0], "").replace(whitespace_unicodes[1], "").strip(" ")  # quick and dirty
                )
            }
        else:
            allowed_tokens = {val: key for key, val in tokenizer.vocab.items() if all(kchr.isalpha() for kchr in key)}
        self.register_buffer(
            "allowed_tokens", torch.as_tensor(sorted(self._apply_blocklist(list(allowed_tokens.keys())))), persistent=False
        )

    @property
    def set(self):
        return self.allowed_tokens


class _GenericNonUniformConstraint(_GenericConstraint):
    """base class with helpers for non-uniform constraints."""

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)

    @property
    def is_uniform(self):
        return False

    def set_at_idx(self, position_idx):
        """For all `num_tokens` positions, which elements are in the set of admissible tokens at this position?"""
        return self.set[position_idx]

    def __len__(self) -> int:
        return max([len(subset) for subset in self.set])

    def select_topk(self, scores, k=5):
        """Return top-k admissible elements based on given scores."""
        return torch.nested.as_nested_tensor(
            [subset[score[subset].topk(min(k, len(subset)), dim=-1).indices] for score, subset in zip(scores, self.set)]
        )

    @staticmethod
    def gather_random_element(indices, locations):
        """Return random element from indices at location."""
        return torch.stack([random.choice(indices[location]) for location in locations])

    def draw_random_sequence(self, device=torch.device("cpu")):
        prompt_ids = torch.cat([subset[torch.randint(len(subset), (1,), device=device)] for subset in self.set])
        # Start with tokenizable prompt
        prompt_ids = self.project_onto_tokenizable_ids(prompt_ids[None])
        return prompt_ids

    @torch.no_grad()
    def normalized_project(self, embedded_inputs, topk=1):
        """if topk > 1, project onto one of the top-k entries randomly."""
        raise NotImplementedError("todo")


class SimpleZalgoConstraint(_GenericNonUniformConstraint):
    """A more complicated constraint. 2-3 tokens are needed for a single zalgo unicode character. These tokens are fixed."""

    zalgos = {
        "up": [
            # these are unicode ords
            *[781, 782, 772, 773, 831, 785, 774, 784, 850, 855, 849, 775, 776, 778, 834, 835, 836, 842, 843, 844, 771, 770, 780, 848, 768],
            *[769, 779, 783, 786, 787, 788, 829, 777, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 830, 859, 838, 794],
        ],
        "middle": [789, 795, 832, 833, 856, 801, 802, 807, 808, 820, 821, 822, 847, 860, 861, 862, 863, 864, 866, 824, 823, 865, 1161],
        "down": [
            *[790, 791, 792, 793, 796, 797, 798, 799, 800, 804, 805, 806, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 825],
            *[826, 827, 828, 837, 839, 840, 841, 845, 846, 851, 852, 853, 854, 857, 858, 803],
        ],
    }

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[], skeleton="please", group_size=2, deconstruct_skeleton=True):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)
        self.group_size = group_size

        if deconstruct_skeleton:
            skeleton_ids = [tokenizer.vocab[s] for s in skeleton]
            skeleton_ids[0] = tokenizer.vocab[whitespace_unicodes[0] + skeleton[0]]  # why llama????????
        else:
            skeleton_ids = tokenizer(skeleton, add_special_tokens=False)["input_ids"]
        num_token_fits_test = (num_tokens - len(skeleton_ids)) % (group_size * len(skeleton_ids))
        if num_token_fits_test != 0:
            raise ValueError(f"Token count {num_tokens} does not fit this constraint. Add or subtract {num_token_fits_test}?")

        # Discover codepoints in tokenizer
        codepoints = []
        for group in self.zalgos:
            for entry in self.zalgos[group]:
                # encode with whitespace for safety in llama:
                token_ids = tokenizer(f" {chr(entry)}", add_special_tokens=False)["input_ids"][1:]
                if len(token_ids) == group_size:
                    codepoints.append(token_ids)

        if len(codepoints) == 0:
            raise ValueError(f"No codepoints of desired group size {group_size} found. Choose a smaller one.")

        valid_codepoints = []
        if group_size == 1:
            valid_codepoints = codepoints
            self.fixed_codes = []
        else:
            # Find the largest group that can be used as uniform constraint:
            counter = defaultdict(int)
            for codepoint in codepoints:
                counter[tuple(codepoint[:-1])] += 1
            max_fixed_code = max(counter, key=counter.get)
            valid_codepoints = [c[-1] for c in codepoints if tuple(c[:-1]) == max_fixed_code]

        self._construct_set(valid_codepoints, max_fixed_code, skeleton_ids)

    def _construct_set(self, valids, fixed_code, skeleton_ids, deconstruct_skeleton=True):
        groups_per_skeleton_char = (self.num_tokens - len(skeleton_ids)) // self.group_size // len(skeleton_ids)
        sets_per_idx = []
        for skeleton_id in skeleton_ids:
            sets_per_idx.append([skeleton_id])
            for group in range(groups_per_skeleton_char):
                for fixed_codepoint in fixed_code:
                    sets_per_idx.append([fixed_codepoint])
                sets_per_idx.append(valids)  # the only part where there can be any optimization happening
        nested_set = torch.nested.nested_tensor(sets_per_idx, dtype=torch.long)
        self.register_buffer("nested_set", nested_set, persistent=False)

    @property
    def set(self):
        return self.nested_set


class ZalgoSkull(SimpleZalgoConstraint):
    skull = r"""                              .___.
          /)               ,-^     ^-.
         //               /           \
.-------| |--------------/  __     __  \-------------------.__
|WMWMWMW| |>>>>>>>>>>>>> | />>\   />>\ |>>>>>>>>>>>>>>>>>>>>>>:>
`-------| |--------------| \__/   \__/ |-------------------'^^
         \\               \    /|\    /
          \)               \   \_/   /
                            |       |
                            |+H+H+H+|
                            \       /
                             ^-----^"""

    def __init__(self, tokenizer, embedding, num_tokens=438, blocklist=[]):
        return super().__init__(tokenizer, embedding, num_tokens, blocklist, skeleton=self.skull, deconstruct_skeleton=False)


class RunicConstraint(_GenericNonUniformConstraint):
    """Elder runes constraint"""

    rune_range = [int("16A0", 16), int("16F8", 16)]

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[], assumed_group_size=3):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)
        num_token_fits_test = (num_tokens % assumed_group_size) - 1
        if num_token_fits_test != 0:
            raise ValueError(f"Token count {num_tokens} does not fit this constraint. Add {num_token_fits_test} more.")
        # Discover codepoints in tokenizer
        codepoints = []
        for entry in range(self.rune_range[0], self.rune_range[1]):
            # encode with whitespace for safety in llama:
            token_ids = tokenizer(f" {chr(entry)}", add_special_tokens=False)["input_ids"][1:]
            codepoints.append(token_ids)

        assert all(len(c) == assumed_group_size for c in codepoints)  # should be true for all normal BPE vocabs

        # Find overlapping groups that can be used as fixed codes:
        counter = defaultdict(int)
        for codepoint in codepoints:
            counter[tuple(codepoint[:-1])] += 1
        fixed_codes = list(counter.keys())
        valid_endpoints = dict(
            zip(counter.keys(), [[c[-1] for c in codepoints if tuple(c[:-1]) == fixed_code] for fixed_code in fixed_codes])
        )

        sets_per_idx = []
        sets_per_idx.append([tokenizer.vocab["▁"]])  # llammaaaaaaaaaaaaaaa
        for group in range(num_tokens // assumed_group_size):
            selected_codepoint = random.choices(list(counter.keys()), weights=counter.values())[0]
            for fixed_codepoint in selected_codepoint:
                sets_per_idx.append([fixed_codepoint])
            sets_per_idx.append(valid_endpoints[selected_codepoint])  # the only part where there can be any optimization happening
        nested_set = torch.nested.nested_tensor(sets_per_idx, dtype=torch.long)
        self.register_buffer("nested_set", nested_set, persistent=False)

    @property
    def set(self):
        return self.nested_set


class CuneiroformConstraint(RunicConstraint):
    """cuneiroform constraint"""

    rune_range = [int("12000", 16), int("123FF", 16)]

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__(tokenizer, embedding, num_tokens, blocklist, assumed_group_size=4)


class HieroglyphConstraint(RunicConstraint):
    """hieroglyphs constraint"""

    rune_range = [int("13000", 16), int("1342F", 16)]

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__(tokenizer, embedding, num_tokens, blocklist, assumed_group_size=4)


class EmojiConstraint(RunicConstraint):
    """EmojiConstraint constraint"""

    rune_range = [int("1F600", 16), int("1F64F", 16)]
    # 2nd range (misc symbols:)
    # U+1F300 to U+1F5FF

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__(tokenizer, embedding, num_tokens, blocklist, assumed_group_size=4)


class InvisibleTagConstraint(RunicConstraint):
    """Based on Riley's twitter comment about invisible unicode via the flag tag."""

    rune_range = [0xE0000 + 64, 0xE0000 + 128]

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__(tokenizer, embedding, num_tokens, blocklist, assumed_group_size=4)


class FullChineseConstraint(RunicConstraint):
    """Sweeping the main chinese unicode range for 3-token combinations"""

    rune_range = [int("4E00", 16), int("9FFF", 16)]

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[], group_size=3):
        _GenericNonUniformConstraint.__init__(self, tokenizer, embedding, num_tokens, blocklist)
        num_token_fits_test = (num_tokens % group_size) - 1
        if num_token_fits_test != 0:
            raise ValueError(f"Token count {num_tokens} does not fit this constraint. Add {num_token_fits_test} more.")
        # Discover codepoints in tokenizer
        codepoints = []
        for entry in range(self.rune_range[0], self.rune_range[1]):
            # encode with whitespace for safety in llama:
            token_ids = tokenizer(f" {chr(entry)}", add_special_tokens=False)["input_ids"][1:]
            if len(token_ids) == group_size:
                codepoints.append(token_ids)

        if len(codepoints) == 0:
            raise ValueError(f"No codepoints of desired group size {group_size} found. Choose a smaller one.")

        # Find overlapping groups that can be used as fixed codes:
        counter = defaultdict(int)
        for codepoint in codepoints:
            counter[tuple(codepoint[:-1])] += 1
        fixed_codes = list(counter.keys())
        valid_endpoints = dict(
            zip(counter.keys(), [[c[-1] for c in codepoints if tuple(c[:-1]) == fixed_code] for fixed_code in fixed_codes])
        )

        sets_per_idx = []
        sets_per_idx.append([tokenizer.vocab["▁"]])  # llammaaaaaaaaaaaaaaa
        for group in range(num_tokens // group_size):
            selected_codepoint = random.choices(list(counter.keys()), weights=counter.values())[0]
            for fixed_codepoint in selected_codepoint:
                sets_per_idx.append([fixed_codepoint])
            sets_per_idx.append(valid_endpoints[selected_codepoint])  # the only part where there can be any optimization happening
        nested_set = torch.nested.nested_tensor(sets_per_idx, dtype=torch.long)
        self.register_buffer("nested_set", nested_set, persistent=False)


class InvisibleConstraint(_GenericNonUniformConstraint):
    """Collect invisible unicodes."""

    invisibles = [
        *[9, 32, 160, 173, 847, 1564, 4447, 4448, 6068, 6069, 6158, 8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199, 8200, 8201, 8202, 8203],
        *[8204, 8205, 8206, 131087, 8239, 8287, 8288, 8289, 8290, 8291, 8292, 8298, 8299, 8300, 8302, 8303, 12288, 10240, 12644, 65279],
        *[65440, 119129, 119155, 119156, 119157, 119158, 119159, 119160, 119161, 119162],
        *[65532, 113824, 113825, 113826, 113827],  # more extreme ones
        *[10, 11, 12, 15, 19, 20, 21, 8233],  # line breaks and separators
    ]

    def __init__(self, tokenizer, embedding, num_tokens=8, blocklist=[]):
        super().__init__(tokenizer, embedding, num_tokens, blocklist)

        # sort invisibles into groups according to tokenizer:
        codepoints = defaultdict(list)
        for entry in self.invisibles:
            # encode with whitespace for safety in llama:
            token_ids = tokenizer(f" {chr(entry)}", add_special_tokens=False)["input_ids"][1:]
            codepoints[len(token_ids)].append(token_ids)

        raise NotImplementedError()
        # todo: choose appropriate codepoints for maximal coverage?
