from matbench.bench import MatbenchBenchmark

mb = MatbenchBenchmark(autoload=False)
mb = mb.from_preset('matbench_v0.1', 'structure')

for task in mb.tasks:
    task.load()
    classification = task.metadata['task_type']
    name = task.dataset_name
    input = task.metadata['input_type']

    for fold in task.folds:
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        
