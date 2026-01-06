#!/bin/bash

# GitHub 仓库设置脚本
# 使用方法: ./setup_github.sh <your-github-repo-url>

set -e

echo "🚀 开始设置 GitHub 仓库..."

# 检查参数
if [ -z "$1" ]; then
    echo "❌ 错误: 请提供 GitHub 仓库地址"
    echo "使用方法: ./setup_github.sh https://github.com/username/repo-name.git"
    exit 1
fi

REPO_URL=$1

# 1. 初始化 Git 仓库
echo "📦 初始化 Git 仓库..."
git init

# 2. 添加所有文件
echo "📝 添加文件到 Git..."
git add .

# 3. 创建首次提交
echo "💾 创建首次提交..."
git commit -m "Initial commit: 基于心理咨询师数字孪生数据集的Qwen3-4B微调项目"

# 4. 添加远程仓库
echo "🔗 添加远程仓库..."
git remote add origin "$REPO_URL" || git remote set-url origin "$REPO_URL"

# 5. 设置默认分支为 main
echo "🌿 设置分支为 main..."
git branch -M main

# 6. 显示状态
echo ""
echo "✅ Git 仓库设置完成！"
echo ""
echo "📊 当前状态:"
git status
echo ""
echo "🔗 远程仓库:"
git remote -v
echo ""
echo "📤 下一步: 推送到 GitHub"
echo "   执行: git push -u origin main"
echo ""

