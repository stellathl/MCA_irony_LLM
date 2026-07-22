import torch


def get_option_confidence(model, tokenizer, formatted_prompt, generated_text, 
                           option_letters=("a", "b", "c", "d"), 
                           forcing_string="\nFinal answer:"):
    """
    Given a prompt + the model's already-generated reasoning trace, force 
    the model to the answer position and extract a softmax distribution 
    over the option letters at that next-token position.
    """
    context_text = formatted_prompt + generated_text + forcing_string

    inputs = tokenizer(context_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]  # logits for the next token

    # Map each letter to its token id — try leading-space variant first,
    # since that's how it'll usually appear after "Final answer:"
    letter_token_ids = {}
    for letter in option_letters:
        ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        letter_token_ids[letter] = ids[0]  # first sub-token

    option_logits = torch.tensor(
        [next_token_logits[letter_token_ids[l]].item() for l in option_letters]
    )
    probs = torch.softmax(option_logits, dim=0)

    return {l: probs[i].item() for i, l in enumerate(option_letters)}, {t: next_token_logits[letter_token_ids[t]].item() for t in option_letters}