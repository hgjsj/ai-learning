import torch
from jinja2.optimizer import optimize


def main():
    a = torch.tensor([2,3], dtype=torch.float32, requires_grad=True)
    b = torch.tensor([6,4], dtype=torch.float32, requires_grad=True)
    Q = a**2 - b**2
    extern_grad = torch.tensor([1.0, 1.0])
    Q.backward(gradient=extern_grad)
    # loss_fn = torch.nn.MSELoss(reduction='mean')
    # loss = loss_fn(a, b)
    # loss.backward()
    # mse = ((b -a) ** 2).mean()
    # print(mse)
    print(a.grad)
    print(b.grad)
    print(2 * a)
    print(-2 * b)
    # optimizer = torch.optim.SGD([a], lr=0.01)
    # optimizer.step()
    # print(a)


if __name__ == "__main__":
    main()
