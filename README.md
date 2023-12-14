

# TransPlace

## For reproduction

1.Get the netlist data in `benchmark/`. the file structure should be like `benchmark/{ispd2015/ispd2019/iccad2015.ot}/{netlist_name}`

2.run `Install.sh` to install dependency

3.run `script_train` to train our model

```bash
python dreamplace/script_train.py --name {your_model_name}
```

the model will be saved in the folder `model` and its name is `your_model_name`. Our train model is available at 

链接：https://pan.baidu.com/s/1bkAKX-BGpJcZH57S46t1gw?pwd=hxor 
提取码：hxor

4.run `script_run_ours.py`

```bash
python dreamplace/script_run_ours.py --name {result_name} --model {your_model_name}
```

the placement will be saved in the `result/{result_name}/{netlist_name}`. You can use `OpenROAD` or other EDA tools to get routability, power, and timing result.