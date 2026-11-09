import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class convolutional_layer:
    def __init__(self, num_filters, kernel_size = 3):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        
        self.filters = np.random.randn(num_filters, kernel_size, kernel_size) / kernel_size ** 2

    def iterate_regions(self, image):
        h, w = image.shape
        k = self.kernel_size 

        for i in range(h - k + 1):
            for j in range(w - k + 1):
                im_region = image[i:i+k, j:j+k]
                yield im_region, i, j
    
    def forward(self, input):
        self.last_input = input
        k = self.kernel_size

    #         out_h = h - k + 1
    # out_w = w - k + 1

    # output = np.zeros((out_h, out_w, self.num_filters))

    # # Сохраняем окна для обратного прохода.
    # self.last_regions = np.zeros(
    #     (out_h, out_w, k, k),
    #     dtype=input.dtype
    # )

    # self.last_relu_mask = np.zeros(output.shape, dtype=bool)

    # # положения окна на изображении.
    # for i in range(out_h):
    #     for j in range(out_w):

    #         # пиксели внутри окна.
    #         for a in range(k):
    #             for b in range(k):
    #                 pixel = input[i + a, j + b]

    #                 self.last_regions[i, j, a, b] = pixel

    #                 #  каждый фильтр.
    #                 for f in range(self.num_filters):
    #                     output[i, j, f] += (
    #                         pixel * self.filters[f, a, b]
    #                     )

    #         #  ReLU к результату каждого фильтра.
    #         for f in range(self.num_filters):
    #             if output[i, j, f] > 0:
    #                 self.last_relu_mask[i, j, f] = True
    #             else:
    #                 output[i, j, f] = 0

        self.last_regions = sliding_window_view(input, (k, k))
        output = np.einsum('ijab,fab->ijf', self.last_regions, self.filters)

        self.last_relu_mask = output > 0
        return output * self.last_relu_mask

    def backprob(self, dl_dout, learn_rate):
        # градиент через ReLU: там, где выход был <= 0, градиент = 0
        dl_dout = dl_dout * self.last_relu_mask

        # dL/dfilter[f] = сумма по (i, j) dL/dout[i, j, f] * окно[i, j]
        dl_dfilters = np.einsum('ijab,ijf->fab', self.last_regions, dl_dout)
        self.filters -= learn_rate * dl_dfilters

        return None
