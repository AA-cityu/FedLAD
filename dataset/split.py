def split_iid(x, y, num_clients):
    """
    Randomly distribute evenly to each client
    """
    from sklearn.utils import shuffle
    x, y = shuffle(x, y)
    shard_size = len(x) // num_clients
    client_data = []
    for i in range(num_clients):
        start = i * shard_size
        end = start + shard_size
        client_data.append((x[start:end], y[start:end]))
    return client_data


def split_noniid(x, y, num_clients):
    """
    Inconsistent label distribution: Shards will be created based on labels (simulating Non-IID)
    """
    from collections import defaultdict
    label_dict = defaultdict(list)
    for xi, yi in zip(x, y):
        label_dict[yi].append(xi)

    client_data = []
    labels = list(label_dict.keys())
    for i in range(num_clients):
        label = labels[i % len(labels)]
        x_selected = label_dict[label][:len(label_dict[label]) // num_clients]
        y_selected = [label] * len(x_selected)
        client_data.append((x_selected, y_selected))
    return client_data
