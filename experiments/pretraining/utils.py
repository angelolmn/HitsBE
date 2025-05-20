import torch

def train_one_epoch(model_engine, dataloader, epoch):
    model_engine.train()

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

        """
        Si voy a usar gradiente acumulado
        loss = loss / model_engine.gradient_accumulation_steps()
        model_engine.backward(loss)

        if model_engine.is_gradient_accumulation_boundary():
            model_engine.step()
        """
        # Calculamos la pérdida 
        loss = torch.nn.functional.cross_entropy(logits.float(), solutions.long())
        loss = loss / model_engine.gradient_accumulation_steps()
        # Reemplaza loss.backward() para usar optimizacion de memoria
        # ZeRO optimization, gradient partitioning, etc.
        model_engine.backward(loss)
        # Reemplaza optimizer.step()
        # sincroniza y actualiza parámetros distribuidos
        model_engine.step()

        predicted_index = torch.argmax(logits, dim=1)
        correct = (predicted_index == solutions).sum().item()
        total = solutions.size(0)
        accuracy = correct / total
        print(f"Epoch {epoch} | Step {model_engine.global_steps} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2%}")

        # Checkpoints automaticos cada X pasos
        if step % model_engine.gradient_accumulation_steps() == 0:
            model_engine.save_checkpoint("checkpoints/", tag=f"epoch{epoch}_step{step}")

        print(f"Step {step} | Acc Boundary: {model_engine.is_gradient_accumulation_boundary()}")

