# JSON标注格式转换工具使用说明

本工具用于将labelme格式的JSON标注文件转换为SAM2训练所需的RLE编码格式。

## 功能特性

- 支持将多边形标注转换为RLE编码
- 批量处理整个目录的JSON文件
- 自动计算掩码面积和边界框
- 保留原始标注类别信息

## 安装依赖

在运行转换脚本前，请确保已安装以下依赖：

```bash
pip install pycocotools numpy pillow
```

## 使用方法

### 单个文件转换

```bash
python convert_json_annotations.py --input dataset/json/1_0-11.json --output dataset/converted_json
```

### 批量转换目录

```bash
python convert_json_annotations.py --input dataset/json --output dataset/converted_json
```

## 输出格式说明

转换后的JSON文件格式如下：

```json
{
  "image": "1_0-11.bmp",  // 图像文件名
  "width": 520,           // 图像宽度
  "height": 520,          // 图像高度
  "annotations": [        // 标注列表
    {
      "id": 1,
      "category_id": 1,
      "category_name": "yijiao",  // 原始标签名
      "segmentation": {           // RLE编码的分割掩码
        "size": [520, 520],
        "counts": "...]"        // RLE字符串
      },
      "area": 928.0,              // 掩码面积
      "bbox": [129.0, 65.0, 54.0, 40.0],  // 边界框 [x, y, width, height]
      "iscrowd": 0
    }
  ]
}
```

## 配置训练数据加载器

转换完成后，需要修改训练配置文件以使用转换后的JSON格式。

### 1. 修改 `bmp_raw_dataset.py`

确保 `MultipleJSONSegmentLoader` 类能够正确加载新的JSON格式。示例实现如下：

```python
class MultipleJSONSegmentLoader:
    def __init__(self, json_dir, img_dir):
        self.json_dir = json_dir
        self.img_dir = img_dir
        
    def load(self, video_name, obj_ids=None):
        json_path = os.path.join(self.json_dir, f"{video_name}.json")
        if not os.path.exists(json_path):
            return [], []
            
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        segments = []
        object_ids = []
        
        for ann in data.get('annotations', []):
            # 确保RLE格式正确
            rle = ann['segmentation']
            # 复制字典避免修改原始数据
            rle_copy = dict(rle)
            # 处理不同格式的counts
            if isinstance(rle_copy['counts'], str):
                # 保持字符串格式
                pass
            elif isinstance(rle_copy['counts'], list):
                # 如果是整数列表，转换为pycocotools需要的格式
                rle_copy['counts'] = bytes(str(rle_copy['counts']), 'utf-8')
                
            segments.append(rle_copy)
            object_ids.append(ann['id'])
            
        return segments, object_ids
```

### 2. 修改训练配置文件

编辑 `training/image_training_config.yaml` 文件：

```yaml
video_dataset:
  _target_: training.dataset.bmp_raw_dataset.BMPRawDataset
  data_root: "D:\\sam2-main\\dataset"
  images_dir: "images"
  json_dir: "converted_json"  # 使用转换后的JSON目录
  image_size: 1024
  # 其他配置保持不变
```

## 注意事项

1. 确保JSON文件与对应的图像文件名称匹配
2. 转换后的RLE编码使用字符串格式，符合SAM2训练要求
3. 对于小面积的标注，训练时可能会被过滤掉，可调整数据加载器中的过滤阈值
4. 建议在转换完成后，随机检查几个文件的格式是否正确

## 故障排除

- **缺少依赖错误**：请确保已安装所有必要的依赖
- **找不到图像文件**：检查图像路径是否正确，或确保JSON文件中包含正确的图像尺寸信息
- **转换失败**：检查JSON文件格式是否符合labelme标准格式

## 示例

转换前的labelme格式：
```json
{
  "version": "0.3.3",
  "shapes": [
    {
      "label": "yijiao",
      "points": [[182.0, 67.0], ...],
      "shape_type": "polygon"
    }
  ],
  "imageWidth": 520,
  "imageHeight": 520,
  "imagePath": "1_0-11.bmp"
}
```

转换后的SAM2训练格式：
```json
{
  "image": "1_0-11.bmp",
  "width": 520,
  "height": 520,
  "annotations": [
    {
      "id": 1,
      "category_name": "yijiao",
      "segmentation": {"size": [520, 520], "counts": "...]"},
      "area": 928.0,
      "bbox": [129.0, 65.0, 54.0, 40.0]
    }
  ]
}
```
