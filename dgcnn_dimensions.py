def forward(self, x):
    # x: bs, 3, np

    # bs 3 N(128)   bs C(224)128 N(128)
    coor = x
    f = self.input_trans(x)
    print(f.size())         # torch.Size([B, 8, 2048])
    print(coor.size())      # torch.Size([B, 3, 2048])
    f = self.get_graph_feature(coor, f, coor, f)
    print(f.size())         # torch.Size([B, 16, 2048, 16])
    f = self.layer1(f)
    f = f.max(dim=-1, keepdim=False)[0]
    print(f.size())         # torch.Size([B, 32, 2048])
    coor_q, f_q = self.fps_downsample(coor, f, 512)
    print(f_q.size())       # torch.Size([B, 32, 512])
    print(coor_q.size())    # torch.Size([B, 3, 512])
    f = self.get_graph_feature(coor_q, f_q, coor, f)
    print(f.size())         # torch.Size([B, 64, 512, 16])
    f = self.layer2(f)
    f = f.max(dim=-1, keepdim=False)[0]
    print(f.size())         # torch.Size([B, 64, 512])
    coor = coor_q

    f = self.get_graph_feature(coor, f, coor, f)
    print(f.size())         # torch.Size([B, 128, 512, 16])
    f = self.layer3(f)
    f = f.max(dim=-1, keepdim=False)[0]
    print(f.size())         # torch.Size([B, 64, 512])

    coor_q, f_q = self.fps_downsample(coor, f, 128)
    print(f_q.size())       # torch.Size([B, 64, 128])
    print(coor_q.size())    # torch.Size([B, 3, 128])
    f = self.get_graph_feature(coor_q, f_q, coor, f)
    print(f.size())         # torch.Size([B, 128, 128, 16])
    f = self.layer4(f)
    f = f.max(dim=-1, keepdim=False)[0]
    coor = coor_q
    print(f.size())         # torch.Size([B, 128, 128])
    print(coor.size())      # torch.Size([B, 3, 128])
    return coor, f


