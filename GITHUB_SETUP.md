# GitHub 上传指南

## 步骤 1: 检查 Git 配置

首先确保你已经配置了 Git 用户信息：

```bash
git config --global user.name "Yanze Jiang"
git config --global user.email "your-email@example.com"
```

## 步骤 2: 初始化 Git 仓库

在项目根目录执行：

```bash
cd /Users/jiangyanze/AI/PsyDTCorpus
git init
```

## 步骤 3: 添加文件

```bash
# 查看将要添加的文件（确认 .gitignore 生效）
git status

# 添加所有文件（.gitignore 会自动排除大文件）
git add .

# 再次查看状态，确认只添加了需要的文件
git status
```

## 步骤 4: 创建首次提交

```bash
git commit -m "Initial commit: 基于心理咨询师数字孪生数据集的Qwen3-4B微调项目"
```

## 步骤 5: 在 GitHub 上创建仓库

1. 访问 https://github.com/new
2. 填写仓库信息：
   - **Repository name**: `PsyDTCorpus-Qwen3-4B-Finetune` (或你喜欢的名字)
   - **Description**: `基于心理咨询师数字孪生数据集的Qwen3-4B微调项目`
   - **Visibility**: 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"（因为我们已经有了）
3. 点击 "Create repository"

## 步骤 6: 连接本地仓库到 GitHub

GitHub 创建仓库后，会显示一个页面，复制仓库地址（例如：`https://github.com/yourusername/PsyDTCorpus-Qwen3-4B-Finetune.git`）

然后执行：

```bash
# 添加远程仓库（替换为你的实际仓库地址）
git remote add origin https://github.com/yourusername/PsyDTCorpus-Qwen3-4B-Finetune.git

# 验证远程仓库
git remote -v
```

## 步骤 7: 推送到 GitHub

```bash
# 如果使用 main 分支
git branch -M main
git push -u origin main

# 或者如果使用 master 分支
git branch -M master
git push -u origin master
```

## 步骤 8: 验证

访问你的 GitHub 仓库页面，确认所有文件都已成功上传。

---

## 注意事项

1. **大文件已被排除**：`.gitignore` 文件已经配置为排除：
   - 模型文件（`.safetensors`, `.bin`, `.pt` 等）
   - 数据集文件（`.json`）
   - 训练检查点

2. **如果需要上传模型文件**：建议使用 Git LFS（Large File Storage）：
   ```bash
   git lfs install
   git lfs track "*.safetensors"
   git lfs track "*.bin"
   git add .gitattributes
   git commit -m "Add Git LFS tracking"
   ```

3. **数据集文件**：由于数据集文件较大，建议：
   - 使用 GitHub Releases 上传
   - 或在 README 中提供下载链接

## 后续更新

当你有新的更改时：

```bash
git add .
git commit -m "描述你的更改"
git push
```

