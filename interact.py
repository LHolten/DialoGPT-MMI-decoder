import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from config import device_f, device_r, num_samples, MMI_temperature, top_k

torch.set_grad_enabled(False)

tokenizer = GPT2Tokenizer('medium/vocab.json', 'medium/merges.txt')

weights = torch.load('medium/medium_ft.pkl')
# fix misused key value
weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
weights.pop("lm_head.decoder.weight", None)

cfg = GPT2Config.from_json_file('medium/config.json')
model: GPT2LMHeadModel = GPT2LMHeadModel(cfg)
model.load_state_dict(weights)
if device_f == 'cuda':
    model.half()
model.to(device_f)
model.eval()

weights = torch.load('medium/small_reverse.pkl')
# fix misused key value
weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
weights.pop("lm_head.decoder.weight", None)

reverse_model: GPT2LMHeadModel = GPT2LMHeadModel(cfg)
reverse_model.load_state_dict(weights)
if device_r == 'cuda':
    reverse_model.half()
reverse_model.to(device_r)
reverse_model.eval()


end_token = torch.tensor([[50256]], dtype=torch.long)


def _get_response(output_token, past):
    out = torch.tensor([[]], dtype=torch.long, device=device_f)

    while True:
        output_token, past = model.forward(output_token, past=past)
        output_token = output_token[:, -1, :].float()
        indices_to_remove = output_token < torch.topk(output_token, top_k)[0][..., -1, None]
        output_token[indices_to_remove] = -float('Inf')
        output_token = torch.multinomial(F.softmax(output_token, dim=-1), num_samples=1)

        out = torch.cat((out, output_token), dim=1)

        if output_token.item() == end_token.item():
            break

    return out, past


def _score_response(output_token, correct_token):
    inputs = torch.cat((output_token, correct_token), dim=1)
    mask = torch.full_like(output_token, -100, dtype=torch.long)
    labels = torch.cat((mask, correct_token), dim=1)

    loss, _, _ = reverse_model(inputs, labels=labels)

    return -loss.float()


def append_messages(old_list: list, new_list: list, truncate_length=64):
    for message in new_list:
        if message != '':
            input_token = tokenizer.encode(message, return_tensors='pt')
            input_token = torch.cat((input_token, end_token), dim=1)
            old_list.append(input_token)

    if len(old_list) == 0:
        old_list.append(end_token)

    # truncate
    total_length = 0
    for i, message in enumerate(reversed(old_list)):
        total_length += message.shape[1]
        if total_length > truncate_length:
            old_list[:] = old_list[-i:]


def generate_message(message_list: list, focus_last_message=True):
    total_input = torch.cat(message_list, dim=1).to(device_f)
    if focus_last_message:
        total_input_reversed = message_list[-1]
    else:
        total_input_reversed = torch.cat(list(reversed(message_list)), dim=1)

    past = None
    if total_input.shape[1] > 1:
        _, past = model(total_input[:, :-1])

    results = []
    for i in range(num_samples):
        result = _get_response(total_input[:, -1:], past)
        score = _score_response(result[0].to(device_r), total_input_reversed.to(device_r))
        results.append(result + (score,))

    scores = torch.stack([x[2] for x in results], dim=0)
    winner = torch.multinomial(F.softmax(scores / MMI_temperature, dim=0), num_samples=1).item()
    # winner = torch.argmax(scores, dim=0)

    out = results[winner][0]

    return tokenizer.decode(out.tolist()[0], skip_special_tokens=True)


if __name__ == '__main__':
    my_message_list = []
    while True:
        my_message = input('usr >> ')
        append_messages(my_message_list, [my_message])
        my_response = generate_message(my_message_list)
        print('bot >>', my_response)
        append_messages(my_message_list, [my_response])

