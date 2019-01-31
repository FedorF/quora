import torch
import numpy as np

def train_model(model, optim, loss_func, num_epoch, train_dl, test_dl, batch_size, path_to_save_model, verbose_every=1000):

    loss_hist = []
    train_acc_hist = []
    test_acc_hist = []

    for epoch in range(num_epoch):
        for step, batch in enumerate(train_dl, 1):

            output = model(batch.question_text)
            loss = loss_func(output, batch.target)
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_hist.append(loss.data.numpy().item())

            if step % verbose_every == 0:
                output = torch.argmax(output, dim=1)
                train_accuracy = np.mean(output.numpy() == batch.target.numpy())
                train_acc_hist.append(train_accuracy)

                correct = 0
                for val_batch in test_dl:
                    output = model(val_batch.question_text)
                    output = np.argmax(output.detach().numpy(), axis=1)
                    correct += np.sum(output == val_batch.target.numpy())

                test_accuracy = correct / (len(test_dl) * batch_size)
                test_acc_hist.append(test_accuracy)

                print('epoch: {} step: {} / {} loss: {} train_acc: {} val_acc: {}'. \
                      format(epoch, step, int(train_dl.__len__() / batch_size), round(loss.data.numpy().item(), 3),
                             round(train_accuracy, 3),
                             round(test_accuracy, 3)))

    torch.save(model.state_dict(), path_to_save_model)
    return loss_hist, train_acc_hist, test_acc_hist