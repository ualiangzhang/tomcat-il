import torch
import torch.nn as nn
import torch.nn.functional as F


class ToMnet(nn.Module):
    def __init__(self, num_inputs=44, num_outputs=6, hidden_size=128, dropout=0.5, without_charnet=False):
        super(ToMnet, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.without_charnet = without_charnet

        # char net settings
        self.char_embedding1 = nn.Linear(num_inputs, hidden_size)
        self.char_bn1 = nn.BatchNorm1d(hidden_size)
        self.char_embedding2 = nn.Linear(hidden_size, hidden_size)
        self.char_bn2 = nn.BatchNorm1d(hidden_size)

        self.char_lstm = nn.LSTM(hidden_size, hidden_size)
        self.char_pooling = nn.AvgPool1d(3, stride=1, padding=1)
        self.char_output = nn.Linear(hidden_size, hidden_size)

        # mental net settings
        self.mental_embedding1 = nn.Linear(num_inputs, hidden_size)
        self.mental_bn1 = nn.BatchNorm1d(hidden_size)
        self.mental_embedding2 = nn.Linear(hidden_size, hidden_size)
        self.mental_bn2 = nn.BatchNorm1d(hidden_size)

        self.mental_lstm = nn.LSTM(hidden_size, hidden_size)
        self.mental_output = nn.Linear(hidden_size, hidden_size)

        # prediction net settings
        self.prediction_embedding1 = nn.Linear(num_inputs - 1, hidden_size)
        self.prediction_bn = nn.BatchNorm1d(hidden_size)
        self.prediction_embedding2 = nn.Linear(hidden_size, hidden_size)

        if self.without_charnet:
            self.feature_extractor1 = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.feature_extractor1 = nn.Linear(3 * hidden_size, hidden_size)
        self.fe_bn1 = nn.BatchNorm1d(hidden_size)
        self.feature_extractor2 = nn.Linear(hidden_size, hidden_size)
        self.fe_bn2 = nn.BatchNorm1d(hidden_size)
        self.fe_pooling = nn.AvgPool1d(3, stride=1, padding=1)

        self.policy_output = nn.Linear(hidden_size, num_outputs)
        self.reward_output = nn.Linear(hidden_size, 1)

    def forward(self, past_traj, pre_traj, state, len_past_traj, len_pre_traj):
        if not self.without_charnet:
            x = torch.reshape(past_traj, [-1, past_traj.shape[-1]]).float()
            x = F.relu(self.char_bn1(self.char_embedding1(x)))
            x = F.relu(self.char_bn2(self.char_embedding2(x)))
            x = torch.reshape(x, [-1, past_traj.shape[0], x.shape[-1]])
            x = nn.utils.rnn.pack_padded_sequence(x, len_past_traj.to('cpu'), enforce_sorted=False)
            _, (x, _) = self.char_lstm(x)
            x = self.char_output(torch.squeeze(self.char_pooling(x), 0))
            x = torch.mean(x, dim=0)
            e_char = torch.cat(state.shape[0] * [torch.unsqueeze(x, 0)])

        x = torch.reshape(pre_traj, [-1, pre_traj.shape[-1]]).float()
        x = F.relu(self.mental_bn1(self.mental_embedding1(x)))
        x = F.relu(self.mental_bn2(self.mental_embedding2(x)))
        x = torch.reshape(x, [-1, pre_traj.shape[0], x.shape[-1]])
        x = nn.utils.rnn.pack_padded_sequence(x, len_pre_traj.to('cpu'), enforce_sorted=False)
        _, (x, _) = self.mental_lstm(x)
        e_mental = self.mental_output(torch.squeeze(x, 0))

        x = state.float()
        x = F.relu(self.prediction_bn(self.prediction_embedding1(x)))
        e_state = self.prediction_embedding2(x)

        if self.without_charnet:
            concatenated_state = self.dropout(torch.cat((e_mental, e_state), 1))
        else:
            concatenated_state = self.dropout(torch.cat((e_char, e_mental, e_state), 1))

        x = F.relu(self.fe_bn1(self.feature_extractor1(concatenated_state)))
        x = F.relu(self.fe_bn2(self.feature_extractor2(x)))
        x = self.fe_pooling(torch.unsqueeze(x, 0))
        policy_logits = self.policy_output(torch.squeeze(x, 0))
        # policy = torch.softmax(policy)
        reward = self.reward_output(torch.squeeze(x, 0))

        return policy_logits, reward
