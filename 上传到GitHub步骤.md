# 📤 上传到 GitHub 完整步骤

## 第一步：初始化本地 Git 仓库

打开终端（Terminal），执行以下命令：

```bash
# 1. 进入项目目录
cd /Users/jiangyanze/AI/PsyDTCorpus

# 2. 初始化 Git 仓库
git init

# 3. 检查 Git 配置（如果需要配置）
git config --global user.name "Yanze Jiang"
git config --global user.email "your-email@example.com"

# 4. 查看将要添加的文件（确认 .gitignore 生效）
git status
```

## 第二步：添加文件并提交

```bash
# 添加所有文件（.gitignore 会自动排除大文件）
git add .

# 查看将要提交的文件
git status

# 创建首次提交
git commit -m "Initial commit: 基于心理咨询师数字孪生数据集的Qwen3-4B微调项目"
```

## 第三步：在 GitHub 上创建新仓库

### 方法一：通过网页创建（推荐）

1. **访问 GitHub 创建仓库页面**
   - 打开浏览器，访问：https://github.com/new
   - 如果未登录，请先登录你的 GitHub 账号

2. **填写仓库信息**
   - **Repository name**: `PsyDTCorpus-Qwen3-4B-Finetune` （或你喜欢的名字）
   - **Description**: `基于心理咨询师数字孪生数据集的Qwen3-4B微调项目`
   - **Visibility**: 
     - 选择 **Public**（公开，其他人可以看到）
     - 或选择 **Private**（私有，只有你可以看到）
   - ⚠️ **重要**：**不要**勾选以下选项：
     - ❌ "Add a README file"（我们已经有了）
     - ❌ "Add .gitignore"（我们已经有了）
     - ❌ "Choose a license"（可选，稍后可以添加）

3. **点击 "Create repository" 按钮**

4. **复制仓库地址**
   - 创建成功后，GitHub 会显示一个页面
   - 找到仓库地址，类似：`https://github.com/yourusername/PsyDTCorpus-Qwen3-4B-Finetune.git`
   - 或者 SSH 地址：`git@github.com:yourusername/PsyDTCorpus-Qwen3-4B-Finetune.git`
   - **复制这个地址**（下一步要用）

### 方法二：通过 GitHub CLI 创建（如果已安装）

```bash
# 安装 GitHub CLI（如果还没安装）
# macOS: brew install gh
# 或访问: https://cli.github.com/

# 登录 GitHub
gh auth login

# 创建仓库
gh repo create PsyDTCorpus-Qwen3-4B-Finetune --public --description "基于心理咨询师数字孪生数据集的Qwen3-4B微调项目"
```

## 第四步：连接本地仓库到 GitHub

回到终端，执行：

```bash
# 添加远程仓库（替换为你在第三步复制的实际地址）
git remote add origin https://github.com/yourusername/PsyDTCorpus-Qwen3-4B-Finetune.git

# 验证远程仓库是否正确添加
git remote -v

# 应该显示：
# origin  https://github.com/yourusername/PsyDTCorpus-Qwen3-4B-Finetune.git (fetch)
# origin  https://github.com/yourusername/PsyDTCorpus-Qwen3-4B-Finetune.git (push)
```

## 第五步：推送到 GitHub

```bash
# 设置默认分支为 main
git branch -M main

# 推送到 GitHub（首次推送）
git push -u origin main

# 如果提示输入用户名和密码：
# - 用户名：你的 GitHub 用户名
# - 密码：需要使用 Personal Access Token（不是登录密码）
#   生成 Token: https://github.com/settings/tokens
#   权限选择：repo
```

## 第六步：验证上传成功

1. **刷新 GitHub 仓库页面**
   - 访问：`https://github.com/yourusername/PsyDTCorpus-Qwen3-4B-Finetune`
   - 应该能看到所有文件

2. **检查文件**
   - ✅ README.md 应该显示在首页
   - ✅ 所有代码文件应该都在
   - ✅ 大文件（模型、数据集）应该被 .gitignore 排除

## 🎉 完成！

现在你的项目已经在 GitHub 上了！

## 📝 后续更新代码

当你对代码做了修改，想要更新到 GitHub：

```bash
# 1. 查看修改的文件
git status

# 2. 添加修改的文件
git add .

# 3. 提交修改
git commit -m "描述你的修改内容"

# 4. 推送到 GitHub
git push
```

## ❓ 常见问题

### Q1: 提示需要用户名和密码？
**A**: GitHub 从 2021 年开始不再支持密码登录，需要使用 Personal Access Token：
1. 访问：https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 勾选 `repo` 权限
4. 生成后复制 Token
5. 推送时，密码处输入这个 Token

### Q2: 想使用 SSH 而不是 HTTPS？
**A**: 如果你已配置 SSH 密钥：
```bash
# 删除 HTTPS 远程仓库
git remote remove origin

# 添加 SSH 远程仓库（替换为你的用户名和仓库名）
git remote add origin git@github.com:yourusername/PsyDTCorpus-Qwen3-4B-Finetune.git

# 推送
git push -u origin main
```

### Q3: 大文件（模型）也想上传？
**A**: 使用 Git LFS（Large File Storage）：
```bash
# 安装 Git LFS
git lfs install

# 跟踪大文件类型
git lfs track "*.safetensors"
git lfs track "*.bin"

# 提交 .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

### Q4: 想忽略某些文件？
**A**: 编辑 `.gitignore` 文件，添加要忽略的文件或文件夹路径。

---

**需要帮助？** 如果遇到问题，随时问我！

