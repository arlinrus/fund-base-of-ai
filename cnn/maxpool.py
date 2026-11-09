import numpy as np

# class maxpoool whitch return max value from grid
class MaxPoll:
    def iterate_regions(self, image):
        h, w, _ = image.shape
        h_2 = h // 2
        w_2 = w// 2

        for i in range(h_2):
            for j in range(w_2):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def to_blocks(self, image):
        h, w, f = image.shape
        h_2, w_2  = h // 2, w // 2
        x = image[:h_2 * 2, :w_2 * 2]
        return x.reshape(h_2, 2, w_2, 2, f).transpose(0, 2, 4, 1, 3).reshape(h_2, w_2, f, 4)

    def backprob(self, dl_dout):
        h, w, f = self.last_input.shape
        h_2, w_2  = h //2 , w // 2

        # gradients will go to pixel where it were max
        grad_blocks = np.zeros((h_2, w_2, f, 4))
        np.put_along_axis(grad_blocks, self.last_argmax[..., None], dl_dout[..., None], axis=3)

        dl_dinput = np.zeros(self.last_input.shape)
        dl_dinput[:h_2 * 2, :w_2 * 2] = (grad_blocks.reshape(h_2, w_2, f, 2, 2).transpose(0, 3, 1, 4, 2).reshape(h_2 * 2, w_2 * 2, f))
        return dl_dinput
    
    def forward(self, input):
        self.last_input = input
        blocks = self.to_blocks(input)
        self.last_argmax = np.argmax(blocks, axis=3)
        return np.max(blocks, axis=3)     