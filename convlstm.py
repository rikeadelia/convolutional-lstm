import torch
from torch.nn.utils.rnn import PackedSequence
from utils import ConvNorm

#TODO: bikin multilayer
#TODO: fixing input yang tidak batch first
#TODO: mampu menangani PackedSequence
class ConvLstm(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers=1, batch_first=True, w_init_gain='linear'):
        super(ConvLstm, self).__init__()

        # self.input_size = input_dim
        # self.batch_size = batch_size
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.W_i = torch.nn.Parameter(torch.randn(self.hidden_size, 1))
        self.W_f = torch.nn.Parameter(torch.randn(self.hidden_size, 1))
        self.W_o = torch.nn.Parameter(torch.randn(self.hidden_size, 1))

        #biases
        self.b_i = torch.nn.Parameter(torch.randn(self.hidden_size, 1))
        self.b_f = torch.nn.Parameter(torch.randn(self.hidden_size, 1))
        self.b_c = torch.nn.Parameter(torch.randn(self.hidden_size, 1))
        self.b_o = torch.nn.Parameter(torch.randn(self.hidden_size, 1))

    def conv_lstm_cell(self, input_, initial_hidden, initial_cell):
        input_size = input_.size(0)
        input_ = input_.permute(1, 2, 0)
        htm1 = initial_hidden.permute(1, 0) #ht-1
        ctm1 = initial_cell.permute(1, 0) #ct-1

        #expand num of weights to batch size
        W_i = self.W_i.expand(self.hidden_size, input_size)
        W_f = self.W_f.expand(self.hidden_size, input_size)
        W_o = self.W_o.expand(self.hidden_size, input_size)

        #expand num of biases to batch size
        b_i = self.b_i.expand(self.hidden_size, input_size)
        b_f = self.b_f.expand(self.hidden_size, input_size)
        b_c = self.b_c.expand(self.hidden_size, input_size)
        b_o = self.b_o.expand(self.hidden_size, input_size)

        hiddens = []
        # cells = []
        for timestep in input_:
            x_i = ConvNorm(self.hidden_size, self.hidden_size)(timestep.unsqueeze(0)).squeeze(0)
            x_f = ConvNorm(self.hidden_size, self.hidden_size)(timestep.unsqueeze(0)).squeeze(0)
            x_c = ConvNorm(self.hidden_size, self.hidden_size)(timestep.unsqueeze(0)).squeeze(0)
            x_o = ConvNorm(self.hidden_size, self.hidden_size)(timestep.unsqueeze(0)).squeeze(0)

            h_i = ConvNorm(self.hidden_size, self.hidden_size)(htm1.unsqueeze(0)).squeeze(0) #convolution for hidden of ingate
            h_f = ConvNorm(self.hidden_size, self.hidden_size)(htm1.unsqueeze(0)).squeeze(0) #convolution for hidden of forgetgate
            h_c = ConvNorm(self.hidden_size, self.hidden_size)(htm1.unsqueeze(0)).squeeze(0) #convolution for hidden of cellgate
            h_o = ConvNorm(self.hidden_size, self.hidden_size)(htm1.unsqueeze(0)).squeeze(0) #convolution for hidden of outgate

            ingate = torch.sigmoid(x_i + h_i + torch.mul(W_i, ctm1) + b_i)
            forgetgate = torch.sigmoid(x_f + h_f + torch.mul(W_f, ctm1) + b_f)
            ct = torch.mul(forgetgate, ctm1) + ingate + torch.tanh(x_c + h_c + b_c)
            outgate = torch.sigmoid(x_o + h_o + torch.mul(W_o, ct) + b_o)
            ht = torch.mul(outgate, torch.tanh(ct))

            hiddens.append(ht.unsqueeze(0))
            # cells.append(ct.unsqueeze(0))

            htm1 = ht
            ctm1 = ct
        
        hy = torch.cat(hiddens, dim=0) #seq_len, hidden_size, batch
        # cy = torch.cat(cells, dim=0)
        return hy, [ht, ct]

    def permute_hidden(self, hx, permutation):
        # type: (Tuple[Tensor, Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        if permutation is None:
            return hx
        return apply_permutation(hx[0], permutation), apply_permutation(hx[1], permutation)

    def forward(self, x, initials):
        orig_input = x
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            x, batch_sizes, sorted_indices, unsorted_indices = x
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = x.size(0) if self.batch_first else x.size(1)
            sorted_indices = None
            unsorted_indices = None

        # if hx is None:
        #     # num_directions = 2 if self.bidirectional else 1
        #     zeros = torch.zeros(self.num_layers * num_directions,
        #                         max_batch_size, self.hidden_size,
        #                         dtype=x.dtype, device=x.device)
        #     hx = (zeros, zeros)
        # else:
        #     # Each batch of the hidden state should match the input sequence that
        #     # the user believes he/she is passing in.
        #     hx = self.permute_hidden(hx, sorted_indices)

        # self.check_forward_args(x, hx, batch_sizes)
        # if batch_sizes is None:
        #     # result = _VF.lstm(x, hx, self._flat_weights, self.bias, self.num_layers,
        #                     #   self.dropout, self.training, self.bidirectional, self.batch_first)
        #     for i in range(x.size(0)):
        #         hx, cx = conv_lstm_cell(x[i], hx, cx, w_ih, w_hh, b_ih, b_hh)
        # else:
        result = self.conv_lstm_cell(x, initials[0], initials[1])

        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        # if isinstance(orig_input, PackedSequence):
        #     output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        #     return output_packed, self.permute_hidden(hidden, unsorted_indices)
        # else:
        #     return output, self.permute_hidden(hidden, unsorted_indices)
        if self.batch_first:
            output = output.permute(2, 0, 1) #batch, seq_len, hidden_size
        return output, (result[0], result[1])

if __name__ == '__main__':
    convlstm = ConvLstm(256)
    x = torch.ones([12, 10, 256])
    initial_h = torch.zeros([12, 256])
    initial_c = torch.zeros([12, 256])
    y, _ = convlstm(x, [initial_h, initial_c])
    print(y.size())
    test_data = torch.ones([1, 10, 256])
    initial_h_test = torch.zeros([1, 256])
    initial_c_test = torch.zeros([1, 256])
    inference, _ = convlstm(test_data, [initial_h_test, initial_c_test])
    print(inference.size())