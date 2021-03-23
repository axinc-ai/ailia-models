import numpy as np

__all__ = [
    'embedding_concat',
]


def embedding_concat(x, y):
    B, C1, H1, W1 = x.shape
    _, C2, H2, W2 = y.shape

    assert H1 == W1

    s = H1 // H2
    sel = [np.array([i for i in range(i, H1, s)]) for i in range(s)]

    a = np.zeros((B, C1 * s * s, H2 * W2))
    for b in range(B):
        for c in range(C1):
            for i in range(s * s):
                a[b, c * s * s + i, :] = x[
                    b, c, sel[i // s][:, None], sel[i % s]
                ].flatten()
    x = a.reshape((B, C1, -1, H2, W2))
    z = np.zeros((B, C1 + C2, s * s, H2, W2))
    for i in range(s * s):
        z[:, :, i, :, :] = np.concatenate((x[:, :, i, :, :], y), axis=1)
    z = z.reshape((B, -1, H2 * W2))

    _, C3, _ = z.shape
    a = np.zeros((B, C3 // (s * s), H1, W1))
    for b in range(B):
        for c in range(C3 // (s * s)):
            for i in range(s * s):
                x = z[b, c * s * s + i, :].reshape((H2, W2))
                a[
                    b, c, sel[i // s][:, None], sel[i % s]
                ] = x

    return a

#
# def embedding_concat(x, y):
#     import torch
#     import torch.nn.functional as F
#
#     x = torch.from_numpy(x)
#     y = torch.from_numpy(y)
#
#     B, C1, H1, W1 = x.size()
#     _, C2, H2, W2 = y.size()
#     s = int(H1 / H2)
#     x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
#     x = x.view(B, C1, -1, H2, W2)
#     z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
#     for i in range(x.size(2)):
#         z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
#     z = z.view(B, -1, H2 * W2)
#     z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
#
#     z = z.numpy()
#
#     return z
