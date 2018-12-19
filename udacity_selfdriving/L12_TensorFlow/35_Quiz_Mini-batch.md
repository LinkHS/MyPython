Let's use mini-batching to feed batches of MNIST features and labels into a linear model.

Set the batch size and run the optimizer over all the batches with the `batches` function in the "./35_Quiz/helper.py". The recommended batch size is 128. If you have memory restrictions, feel free to make it smaller.

[Quiz](./35_Quiz/Quiz_jupyter.md)

[solution](./35_Quiz/solution.md)

Here is my solution.
```python
n = train_features.shape[0] // batch_size
for i in range(n):
    batch_features = train_features[i*batch_size:(i+1)*batch_size, :]
    batch_labels = train_labels[i*batch_size:(i+1)*batch_size, :]
    sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})
 ```
