import torch

# 把路径换成你报错的那个权重文件路径
pt_path = r'runs\train\yolo11-687-hsfpn+mona1+AYHead\weights\best.pt' # 举例，替换为你的实际路径

print(f"正在加载: {pt_path} ...")
try:
    # 加载 pt 文件
    ckpt = torch.load(pt_path, map_location='cpu')
    
    # 获取模型对象
    # Ultralytics 的 pt 文件通常是一个字典，模型在 'model' 键下
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model = ckpt['model']
    else:
        model = ckpt
    
    print("\n=== 模型加载成功 ===")
    
    # 遍历模型的所有子模块，寻找 ProgressiveTSSA_Fusion
    found = False
    for name, module in model.named_modules():
        # 这里的名字 'ProgressiveTSSA_Fusion' 要和你的类名一致
        if "ProgressiveTSSA_Fusion" in str(type(module)):
            found = True
            print(f"\n找到模块: {name} ({type(module).__name__})")
            print(f"该模块包含的属性 (Attributes):")
            
            # 打印该模块的所有成员变量（子层）
            # 也就是你在 __init__ 里 self.xxx = ... 的那些东西
            for key, value in module.__dict__.items():
                # 过滤掉一些内部自带的属性，只看你定义的层
                if not key.startswith('_') and isinstance(value, (torch.nn.Module, torch.Tensor)):
                    print(f"  - self.{key} : {type(value).__name__}")
            
            print("-" * 30)
            # 如果你想看所有这类模块，去掉下面这行 break
            # break 
            
    if not found:
        print("未在模型中找到名为 ProgressiveTSSA_Fusion 的模块。")

except Exception as e:
    print(f"加载出错: {e}")
    print("提示：如果报错说找不到类，说明你现在的代码改动太大，连旧模型都无法反序列化了。")