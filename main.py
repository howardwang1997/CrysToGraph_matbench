from matbench.bench import MatbenchBenchmark

mb = MatbenchBenchmark(autoload=False)

for task in mb.tasks:
    task.load()
    for fold in task.folds:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
