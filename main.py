import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
import os

def load_image(image_path):
    """加载图像"""
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None

def detect_artifacts(image):
    """检测图像中的融合伪影"""
    # 转换为numpy数组
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    artifacts = []
    
    # 1. 检测重影伪影（双重边缘）
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 过滤小轮廓
            x, y, w, h = cv2.boundingRect(contour)
            # 检查是否有重影特征
            roi = gray[y:y+h, x:x+w]
            if roi.size > 0:
                std_dev = np.std(roi)
                if std_dev > 30:  # 标准差较大可能表示重影
                    artifacts.append({
                        'type': '重影伪影',
                        'bbox': (x, y, w, h),
                        'confidence': min(std_dev/50, 1.0),
                        'color': 'red'
                    })
    
    # 2. 检测条纹伪影
    # 水平条纹检测
    h_kernel = np.ones((1, 15), np.uint8)
    h_morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, h_kernel)
    h_diff = cv2.absdiff(gray, h_morph)
    h_thresh = cv2.threshold(h_diff, 30, 255, cv2.THRESH_BINARY)[1]
    
    # 垂直条纹检测
    v_kernel = np.ones((15, 1), np.uint8)
    v_morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, v_kernel)
    v_diff = cv2.absdiff(gray, v_morph)
    v_thresh = cv2.threshold(v_diff, 30, 255, cv2.THRESH_BINARY)[1]
    
    # 合并条纹检测结果
    stripes = cv2.bitwise_or(h_thresh, v_thresh)
    stripe_contours, _ = cv2.findContours(stripes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in stripe_contours:
        if cv2.contourArea(contour) > 200:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 3:  # 长宽比大于3认为是条纹
                artifacts.append({
                    'type': '条纹伪影',
                    'bbox': (x, y, w, h),
                    'confidence': min(aspect_ratio/10, 1.0),
                    'color': 'orange'
                })
    
    # 3. 检测色彩不一致（仅对彩色图像）
    if len(img_array.shape) == 3:
        # 计算颜色方差
        b, g, r = cv2.split(img_array)
        color_var = np.var([np.mean(b), np.mean(g), np.mean(r)])
        
        if color_var > 1000:  # 颜色方差较大
            # 分块检测色彩不一致
            h, w = gray.shape
            block_size = 50
            for i in range(0, h-block_size, block_size):
                for j in range(0, w-block_size, block_size):
                    block = img_array[i:i+block_size, j:j+block_size]
                    if block.size > 0:
                        block_var = np.var(block, axis=(0, 1))
                        if np.max(block_var) > 500:
                            artifacts.append({
                                'type': '色彩不一致',
                                'bbox': (j, i, block_size, block_size),
                                'confidence': min(np.max(block_var)/1000, 1.0),
                                'color': 'blue'
                            })
    
    # 4. 检测边界模糊
    # 使用拉普拉斯算子检测模糊区域
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = cv2.convertScaleAbs(laplacian)
    
    # 找到模糊区域（拉普拉斯方差较小的区域）
    blur_mask = laplacian_var < 20
    blur_contours, _ = cv2.findContours(blur_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in blur_contours:
        if cv2.contourArea(contour) > 300:
            x, y, w, h = cv2.boundingRect(contour)
            artifacts.append({
                'type': '边界模糊',
                'bbox': (x, y, w, h),
                'confidence': 0.7,
                'color': 'green'
            })
    
    # 5. 检测噪声伪影
    # 使用高斯滤波后的差值检测噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, blurred)
    noise_thresh = cv2.threshold(noise, 15, 255, cv2.THRESH_BINARY)[1]
    
    noise_contours, _ = cv2.findContours(noise_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in noise_contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            artifacts.append({
                'type': '噪声伪影',
                'bbox': (x, y, w, h),
                'confidence': 0.6,
                'color': 'purple'
            })
    
    return artifacts

def annotate_artifacts(image, artifacts):
    """在图像上标注伪影"""
    # 创建可绘制的图像副本
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # 为每种伪影类型分配颜色
    colors = {
        '重影伪影': 'red',
        '条纹伪影': 'orange', 
        '色彩不一致': 'blue',
        '边界模糊': 'green',
        '噪声伪影': 'purple'
    }
    
    artifact_count = {}
    
    for artifact in artifacts:
        x, y, w, h = artifact['bbox']
        artifact_type = artifact['type']
        confidence = artifact['confidence']
        color = colors.get(artifact_type, 'red')
        
        # 绘制边界框
        draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
        
        # 添加标签
        artifact_count[artifact_type] = artifact_count.get(artifact_type, 0) + 1
        label = f"{artifact_type} #{artifact_count[artifact_type]}"
        
        # 绘制标签背景
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.rectangle([x, y-text_height-2, x+text_width+4, y], fill=color)
        draw.text((x+2, y-text_height-1), label, fill='white', font=font)
        
        # 添加置信度
        conf_text = f"({confidence:.2f})"
        draw.text((x+2, y+2), conf_text, fill=color, font=font)
    
    return annotated_image, artifact_count

def create_artifact_analysis_report(artifacts, artifact_count):
    """创建伪影分析报告"""
    report = "=== 融合伪影检测报告 ===\n\n"
    
    if not artifacts:
        report += "未检测到明显的融合伪影。\n"
        return report
    
    report += f"总共检测到 {len(artifacts)} 个伪影区域\n\n"
    
    for artifact_type, count in artifact_count.items():
        report += f"{artifact_type}: {count} 个\n"
        
        # 获取该类型伪影的详细信息
        type_artifacts = [a for a in artifacts if a['type'] == artifact_type]
        avg_confidence = np.mean([a['confidence'] for a in type_artifacts])
        report += f"  - 平均置信度: {avg_confidence:.2f}\n"
        
        # 伪影特征描述
        if artifact_type == '重影伪影':
            report += "  - 特征: 图像中出现双重边缘或重叠效果\n"
            report += "  - 成因: 配准不准确或多源图像融合时的空间偏移\n"
        elif artifact_type == '条纹伪影':
            report += "  - 特征: 图像中出现规律性的条纹模式\n"
            report += "  - 成因: 传感器扫描方式或融合算法的周期性误差\n"
        elif artifact_type == '色彩不一致':
            report += "  - 特征: 不同区域的颜色分布不均匀\n"
            report += "  - 成因: 多光谱图像融合时的光谱响应差异\n"
        elif artifact_type == '边界模糊':
            report += "  - 特征: 图像边缘或细节区域出现模糊\n"
            report += "  - 成因: 融合过程中的空间分辨率损失\n"
        elif artifact_type == '噪声伪影':
            report += "  - 特征: 图像中出现随机的噪声点\n"
            report += "  - 成因: 传感器噪声或融合算法引入的噪声\n"
        
        report += "\n"
    
    # 添加改进建议
    report += "=== 改进建议 ===\n"
    if '重影伪影' in artifact_count:
        report += "• 改进图像配准精度，使用更精确的配准算法\n"
    if '条纹伪影' in artifact_count:
        report += "• 优化融合算法，减少周期性误差\n"
    if '色彩不一致' in artifact_count:
        report += "• 改进光谱匹配和颜色校正算法\n"
    if '边界模糊' in artifact_count:
        report += "• 使用保边融合算法，如基于梯度的融合方法\n"
    if '噪声伪影' in artifact_count:
        report += "• 加强去噪预处理和后处理步骤\n"
    
    return report

def main():
    """主函数"""
    image_path = "11.png"
    
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 找不到图像文件 {image_path}")
        return
    
    print("正在加载图像...")
    image = load_image(image_path)
    
    if image is None:
        print("图像加载失败")
        return
    
    print("正在检测融合伪影...")
    artifacts = detect_artifacts(image)
    
    print(f"检测到 {len(artifacts)} 个潜在伪影区域")
    
    # 标注伪影
    print("正在标注伪影...")
    annotated_image, artifact_count = annotate_artifacts(image, artifacts)
    
    # 保存标注后的图像
    output_path = "11_annotated_artifacts.png"
    annotated_image.save(output_path)
    print(f"标注后的图像已保存为: {output_path}")
    
    # 生成分析报告
    report = create_artifact_analysis_report(artifacts, artifact_count)
    
    # 保存报告
    report_path = "artifact_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"分析报告已保存为: {report_path}")
    
    # 显示结果
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title('原始图像', fontsize=14)
    axes[0].axis('off')
    
    # 标注图
    axes[1].imshow(annotated_image)
    axes[1].set_title(f'伪影标注图 (检测到{len(artifacts)}个伪影)', fontsize=14)
    axes[1].axis('off')
    
    # 添加图例
    legend_elements = []
    colors = {
        '重影伪影': 'red',
        '条纹伪影': 'orange', 
        '色彩不一致': 'blue',
        '边界模糊': 'green',
        '噪声伪影': 'purple'
    }
    
    for artifact_type, count in artifact_count.items():
        color = colors[artifact_type]
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, 
                                           label=f'{artifact_type} ({count})'))
    
    if legend_elements:
        axes[1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('artifact_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印报告摘要
    print("\n" + "="*50)
    print("伪影检测摘要:")
    print("="*50)
    for artifact_type, count in artifact_count.items():
        print(f"{artifact_type}: {count} 个")
    print("="*50)
    print(f"详细报告请查看: {report_path}")

if __name__ == "__main__":
    main()
