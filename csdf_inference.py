import torch
import os
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
from mlp import MLPRegression
from siren_pytorch import SirenNet


MODEL_CHECKPOINT_FILENAME = 'siren_batch2000x100.pt'


def inference(x,q,model):
    model.eval()
    x_cat = x.unsqueeze(1).expand(-1,len(q),-1).reshape(-1,3)
    q_cat = q.unsqueeze(0).expand(len(x),-1,-1).reshape(-1,7)
    inputs = torch.cat([x_cat,q_cat],dim=-1)
    pred = model.forward(inputs)
    return pred

def main_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # CHOOSE CLASSIC MLP, SIREN OR SIREN LIGHT
    # model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)  # MLP
    model = SirenNet(dim_in=10, dim_out=1, num_layers=3, dim_hidden=512)  # SIREN
    # model = SirenNet(dim_in=10, dim_out=1, num_layers=3, dim_hidden=256)  # SIREN light


    # LOAD MODEL
    model.load_state_dict(torch.load(os.path.join(CUR_PATH,MODEL_CHECKPOINT_FILENAME))[49900])
    model.to(device)


    # INPUT
    x = torch.tensor([[0.0, 0.0, 0.0]],requires_grad=True).to(device).float()                       # PROBE POINT (cartesian space)
    q = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],requires_grad=True).to(device).float()   # ROBOT ORIENTATION (radians)

    x,q = x.to(device),q.to(device)

    d = inference(x, q, model)
    print(d)

if __name__ == '__main__':
    main_loop()