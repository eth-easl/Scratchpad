import torch


def test():
    all_logprobs = torch.tensor(
        #       s                     s                s
        [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
        dtype=torch.float32,
        device="cuda",
    )
    seq_lens = torch.tensor([2, 0, 3, 0], dtype=torch.int32, device="cuda")
    input_ids = torch.tensor([1, 2, 3, 0, 1], dtype=torch.int32, device="cuda")

    token_logprobs = all_logprobs[
        torch.arange(all_logprobs.shape[0], device="cuda"),
        torch.cat([input_ids[1:], torch.tensor([0], device="cuda")]),
    ]
    logprobs_cumsum = torch.cumsum(token_logprobs, dim=0, dtype=torch.float32)

    len_cumsum = torch.cumsum(seq_lens, dim=0)
    start = torch.cat((torch.tensor([0], device="cuda"), len_cumsum[:-1]), 0)
    end = start + seq_lens - 2
    start.clamp_(min=0, max=token_logprobs.shape[0] - 1)
    end.clamp_(min=0, max=token_logprobs.shape[0] - 1)
    sum_logp = logprobs_cumsum[end] - logprobs_cumsum[start] + token_logprobs[start]

    # assert logprobs == [2, _, 2, 4, _]
    print("token logprobs", token_logprobs)
    print("start", start)
    print("end", end)
    print("sum_logp", sum_logp)


if __name__ == "__main__":
    test()
