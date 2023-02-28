import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18


# this network correspond to phi network in paper
class FEATURE_EXTRACTOR(nn.Module):
    def __init__(self):
        super(FEATURE_EXTRACTOR, self).__init__()
        self.extractor = resnet18()                                         # out: b, 1024
        self.pose_fc = nn.Linear(5, 16)                                     # x/lam, y/lam, cos(theta), sin(theta), e^-t
        self.action_fc = nn.Linear(1, 16)                                   # action = {0,1,2,3,4,5}

        self.fc = nn.Linear(1024+16+16, 128)

    def forward(self, seq_img, seq_pose, seq_action):
        img_feature = self.extractor(seq_img)
        pose_feature = F.relu(self.pose_fc(seq_pose))
        action_feature = F.relu(self.action_fc(seq_action))

        o = torch.cat([img_feature, pose_feature, action_feature], dim=1)   # o: b, 1024+16+16
        o = F.relu(self.fc(o))
        return o

    def set_weights(self, weights):
        self.load_state_dict(weights)


# This is the Q function, output expected rewards. In section 3.2.2 they used 2 fc at the end
class DQN_POLICY(nn.Module):
    def __init__(self):
        super(DQN_POLICY, self).__init__()
        self.encoder = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)    # self attention between M and M, output C
        self.decoder = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)    # attention between phi(o), C
        self.fc_1 = nn.Linear(128, 32)
        self.fc_2 = nn.Linear(32, 6)                                        # action space in my env is 6

    # M is all prev mem(b,t,128), o is features(b,128)
    def forward(self, M, o, encoder_masks, decoder_masks):
        if encoder_masks is not None:
            C, _ = self.encoder(M, M, M, need_weights=False, attn_mask=encoder_masks)
        else:
            C, _ = self.encoder(M, M, M, need_weights=False)
        o = o.unsqueeze(1)
        if decoder_masks is not None:
            decoder_out, _ = self.decoder(o, C, C, need_weights=False, attn_mask=decoder_masks)
        else:
            decoder_out, _ = self.decoder(o, C, C, need_weights=False)
        decoder_out = self.fc_2(F.relu(self.fc_1(decoder_out)))

        return decoder_out.squeeze(1)

    def set_weights(self, weights):
        self.load_state_dict(weights)


if __name__ == '__main__':
    # test for feature_extractor
    """
    model = FEATURE_EXTRACTOR().cuda()
    img = torch.randn(10, 3, 80, 80).cuda()
    pose = torch.randn(10, 5).cuda()
    action = torch.randn(10, 1).cuda()
    out = model(img, pose, action)
    """
    # test for decoder
    model = DQN_POLICY().cuda()
    m = torch.randn(10, 21, 128).cuda()
    # feature = torch.randn(10, 128).cuda()
    out = model(m)

