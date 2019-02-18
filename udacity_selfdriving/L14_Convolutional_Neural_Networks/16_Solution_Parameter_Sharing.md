**Solution**
There are `3860` total parameters. That's 196 times fewer parameters! Here's how the answer is calculated:

`(8 * 8 * 3 + 1) * 20 = 3840 + 20 = 3860`

That's `3840` weights and `20` biases. This should look similar to the answer from the previous quiz. The difference being it's just `20` instead of (`14 * 14 * 20`). Remember, with weight sharing we use the same filter for an entire depth slice. Because of this we can get rid of `14 * 14` and be left with only `20`.

