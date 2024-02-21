"""
Simple interactive console for people too lazy to fire up llama.cpp (me)
"""
import torch
import carving
import readline

readline.parse_and_bind("tab: complete")


setup = dict(dtype=torch.float16, device=torch.device("cuda:0"))
cfg_impl = dict(load_in_8bit=False, grad_checkpointing=False, use_flash_attention_2=True, use_kv_caching=False)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # Should be true anyway


def get_multi_line_input(prompt="\033[96mUser (press Ctrl+D or empty line to finish):\033[0m"):
    lines = []
    while True:
        try:
            line = input(prompt)
        except EOFError:
            break  # Ctrl+D was pressed
        if line == "":  # no empty lines
            break
        else:
            lines.append(line)
    return "\n".join(lines)


def main_process(setup=setup):
    model_name = input("Provide a model location ...") or "/fs/nexus-scratch/jonas0/llama2-7B-chat"
    model, tokenizer = carving.load_model_and_tokenizer(model_name, cfg_impl, setup)
    model.to(**setup)  # Move all parts to GPU here
    model = torch.compile(model)

    generation_args = dict(use_cache=True, do_sample=True)

    print("------------------------------------------------------------------------------------")
    messages = [{"role": "system", "content": tokenizer.default_system_message}]
    print(tokenizer.default_system_message)

    while True:
        user_message = input("\033[96mUser:\033[0m") or ""
        if user_message == "RESET":
            print("------------------------------------------------------------------------------------")
            messages = [{"role": "system", "content": tokenizer.default_system_message}]
            print(tokenizer.default_system_message)
            continue
        elif user_message == "PASTE":
            import pyperclip

            user_message = pyperclip.paste()
        elif user_message == "REDO":
            messages.pop()
        elif user_message == "MULTILINE":
            user_message = get_multi_line_input()
        else:
            messages += [{"role": "user", "content": user_message}]
        prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        outputs = model.generate(
            torch.as_tensor(prompt_ids, device=setup["device"])[None],
            max_new_tokens=1024,
            pad_token_id=tokenizer.pad_token_id,
            **generation_args,
        )
        outputs = outputs[0, len(prompt_ids) :]
        decoded_outputs = tokenizer.decode(outputs)
        messages += [{"role": "assistant", "content": decoded_outputs}]
        print(f"\u001b[31mAssistant:\033[0m {decoded_outputs}")


if __name__ == "__main__":
    main_process()
