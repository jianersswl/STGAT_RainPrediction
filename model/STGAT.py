import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class SpatioTemporalAttentionGAT_Two(torch.nn.Module):
    def __init__(self, time_dim, nodes_dim, spatio_feature_dim, target_dim, edge_index, gat_hidden_dim=256, encoder_depth=4, heads=4):
        super(SpatioTemporalAttentionGAT_Two, self).__init__()
        
        self.time_dim = time_dim
        self.nodes_dim = nodes_dim
        self.encoder_depth = encoder_depth
        self.spatio_feature_dim = spatio_feature_dim
        self.gat_hidden_dim = gat_hidden_dim
        self.target_dim = target_dim
        self.heads = heads
        self.edge_index = edge_index
        # Spatio encoder
        # 创建卷积层的序列
        self.spatio_encoders = nn.ModuleList()
        for i in range(encoder_depth):
            self.spatio_encoders.append(nn.Conv2d(2**i, 2**(i+1), kernel_size=3, padding=1, stride=2))
            self.spatio_encoders.append(nn.ReLU())
        
        # Temporal encoder
        spatio_neck_dim = self.spatio_feature_dim**2//2**self.encoder_depth
        self.attn_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=spatio_neck_dim, num_heads=self.heads)\
                                          for i in range(self.nodes_dim)])
        
        # Blockneck
        self.gat_conv = GATConv(spatio_neck_dim*self.time_dim, \
                                spatio_neck_dim, \
                                heads=self.heads, concat=True)

        # Reshape_b4_decoder
        self.reshape_b4_decoder = nn.Sequential(
            nn.Linear(spatio_neck_dim * self.heads, spatio_neck_dim),
            nn.ReLU(),
        )
        
        # Spatio decoder
        self.decoder_upconvs = nn.ModuleList()
        for i in range(encoder_depth):
            self.decoder_upconvs.append(nn.ConvTranspose2d(2**(self.encoder_depth-i), 2**(self.encoder_depth-i-1), \
                                                          kernel_size=3, stride=2, padding=1, output_padding=1))
            self.decoder_upconvs.append(nn.ReLU())
            
        self.skip_connections = nn.ModuleList([nn.Conv2d(2**(self.encoder_depth-i)*(self.time_dim*self.nodes_dim+1),\
                                                         2**(self.encoder_depth-i), kernel_size=1) for i in range(self.encoder_depth)])
        
        # Final Conv
        self.final_conv_class = nn.Conv2d(1, self.target_dim, kernel_size=1)  
        self.final_conv_regre = nn.Conv2d(1, 1, kernel_size=1) 
    
    def spatio_extractor(self, x, init_shape):
        B, T, N, H, W = init_shape
        x_spatio = x.view(B * T * N, H, W).unsqueeze(1) # (B, T, N, H, W) ---> (B*T*N, 1, H, W)
        
        spatio_features = []
        for i in range(0, len(self.spatio_encoders), 2):
            conv_layer = self.spatio_encoders[i]     # 获取卷积层
            relu_layer = self.spatio_encoders[i + 1]  # 获取 ReLU 激活层
            x_spatio = conv_layer(x_spatio)          # 应用卷积
            x_spatio = relu_layer(x_spatio)          # 应用 ReLU 激活
#             print(f'x_spatio.shape:{x_spatio.shape}')
            spatio_features.append(x_spatio.view(B, T, N,x_spatio.shape[-3], x_spatio.shape[-2],x_spatio.shape[-1]))

        return spatio_features
    
    def temporal_extractor(self, x_temporal, init_shape):
        B, T, N, H, W = init_shape
        attn_outputs = []
        for n, attn_layer in enumerate(self.attn_layers):
            query = x_temporal[:, :, n, :]
            key = query  
            value = query

            # 调用 MultiheadAttention
            attn_output, attn_weights = attn_layer(query, key, value)

            # 存储每个 V 维度的结果
            attn_outputs.append(attn_output)

        return attn_outputs
    
    # Number of node and edge_index for each graph in the batch should be the same
    def gat_blockneck(self, x, edge_index, init_shape):
        B, T, N, H, W = init_shape
        x_batch = x.reshape(B * N, -1)
        # print(f'attn_outputs_batch.shape:{x_batch.shape}')
        # print(f'edge_index.shape:{edge_index.shape}')
        edge_indices = []
        offset = 0  
        for i in range(B):
            current_edge_index = edge_index + offset
            edge_indices.append(current_edge_index)
            offset += N  # 每个图的节点数是 N
        edge_index_batch = torch.cat(edge_indices, dim=1)
        # print(f'edge_index_batch.shape:{edge_index_batch.shape}')
        node_features = self.gat_conv(x_batch, edge_index_batch) # (B*N, heads*gat_hidden_dim)
#         print(f'gat_node_features.shape:{node_features.shape}')
        node_features = node_features.view(B, N, -1)
#         print(f'gat_node_features.shape:{node_features.shape}')
        node_features = node_features.sum(dim=1)                     # (B, gat_hidden_dim)
#         print(f'sum_node_features.shape:{node_features.shape}')

        return node_features

    def upscaling(self, x, spatio_features, init_shape):
        B, T, N, H, W = init_shape
        final_x = x
        for i in  range(0, len(self.decoder_upconvs), 2):
            # 每次使用两个layer，分别是卷积层和ReLU激活层
            conv_layer = self.decoder_upconvs[i]
            relu_layer = self.decoder_upconvs[i+1]  # 获取对应的ReLU激活层
            
            # Skip connection
#             print(f'final_node_feature.shape:{final_node_feature.shape}')
            skip_passing = spatio_features[len(spatio_features)-1-i//2]
#             print(f'skip_passing.shape:{skip_passing.shape}')
            final_x = torch.cat((final_x, skip_passing.view(B, T*N*skip_passing.shape[-3], \
                                                skip_passing.shape[-2],\
                                                skip_passing.shape[-1])), \
                                           dim=1)
            final_x = self.skip_connections[i//2](final_x)
            # 应用卷积和ReLU激活层
            final_x = conv_layer(final_x)
            final_x = relu_layer(final_x)

        return final_x
    
    def forward(self, x, edge_index):
        init_shape = x.shape
        B, T, N, H, W = init_shape
        
        spatio_features = self.spatio_extractor(x, init_shape)
#         print(spatio_features[-1].shape)
#         print(spatio_features[-1].view(B, T, N, -1).shape)
        x_temporal = spatio_features[-1].view(B, T, N, -1)
        attn_outputs = self.temporal_extractor(x_temporal, init_shape)
#         print(f'x_temporal.shape:{x_temporal.shape}')
        attn_outputs = torch.stack(attn_outputs, dim=0)  # 形状变为 (N, B, T, A)
#         print(f'attn_outputs.shape:{attn_outputs.shape}')
        attn_outputs = attn_outputs.permute(1, 0, 2, 3)  # 转置成 (B, N, T, A)
#         print(f'attn_outputs.shape:{attn_outputs.shape}')
        attn_outputs = attn_outputs.reshape(B, N, -1)  # 将 (B, N, T, A) 转换为 (B, N, T * A)
        # print(f'attn_outputs.shape:{attn_outputs.shape}')
        
        node_features = self.gat_blockneck(attn_outputs, self.edge_index, init_shape)
#         print(f'sum_node_features.shape:{node_features.shape}')
        final_node_feature = self.reshape_b4_decoder(node_features)  # (B, 64 * 64)
#         print(f'b4_decoder_final_node_feature.shape:{final_node_feature.shape}')
        final_node_feature = final_node_feature.view(B,2**(self.encoder_depth), \
                                                     self.spatio_feature_dim//2**(self.encoder_depth), \
                                                     self.spatio_feature_dim//2**(self.encoder_depth))
#         print(f'b4_decoder_final_node_feature.shape:{final_node_feature.shape}')
        
        upscaling_feature = self.upscaling(final_node_feature, spatio_features, init_shape)
#         print(f'up_final_node_feature.shape:{final_node_feature.shape}')

        class_result = self.final_conv_class(upscaling_feature)
        regre_result = self.final_conv_regre(upscaling_feature)
        
        return class_result, regre_result