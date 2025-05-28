import torch

def train_one_epoch(model_engine, dataloader, epoch):
    model_engine.train()
    accuracy = []
    losses = []
    # Step es el numero de lote dentro de la epoca
    for step, batch in enumerate(dataloader):
        # Se mueven los tensores a la GPU correspondiente (local_rank)
        input = batch["input"].to(model_engine.local_rank)     # (B, 3, 1024)
        mask = batch["mask"].to(model_engine.local_rank)       # (B, 128)
        solutions = batch["solution"].to(model_engine.local_rank)     # (B,)

        # X: time series, 3 for each test (batch_size, 3, ts_len)
        # MLM: tensor torch binario con 1 en [MASK] (batch_size, dim_seq)
        # solutions: (batch_size,)
        logits = model_engine(X=input, MLM=mask, solutions=solutions)  # (B, 3)

        # Calculamos la pérdida 
        loss = torch.nn.functional.cross_entropy(logits.float(), solutions.long())
        
        # ZeRO optimization, gradient partitioning, etc.
        model_engine.backward(loss)
        # Reemplaza optimizer.step()
        # sincroniza y actualiza parámetros distribuidos
        model_engine.step()

        predicted_index = torch.argmax(logits, dim=1)
        correct = (predicted_index == solutions).sum().item()
        total = solutions.size(0)
        accuracy.append(correct / total)
        
        losses.append(loss.item())

        #print(logits)
        #print(loss)

        if (step + 1) % model_engine.gradient_accumulation_steps() == 0:
            with open("experiments/pretraining/.steps_training_log.txt", "a") as logfile_step:
            
                loss_mean = torch.mean(torch.tensor(losses))
                accuracy_mean = torch.mean(torch.tensor(accuracy))

                line_step = f"Step {model_engine.global_steps} | Loss: {loss_mean:.2f} | Accuracy: {accuracy_mean:.2%}"
            
                print(line_step, file=logfile_step)
            
                losses = []
                accuracy = []

        # Checkpoints automaticos cada X pasos
        if (step + 1) % (model_engine.gradient_accumulation_steps() * 130) == 0:
            model_engine.save_checkpoint("experiments/pretraining/checkpoints/", tag=f"epoch{epoch}_step{step}")


