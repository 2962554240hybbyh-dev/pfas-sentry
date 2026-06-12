/**
 * PFAS-Sentry 前端应用
 */

// API 基础地址
const API_BASE = '';

/**
 * 通用 API 请求函数
 */
async function apiRequest(endpoint, method = 'GET', data = null) {
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json',
        },
    };

    if (data) {
        options.body = JSON.stringify(data);
    }

    try {
        const response = await fetch(API_BASE + endpoint, options);
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || '请求失败');
        }

        return result;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

/**
 * 获取化合物列表
 */
async function getCompounds() {
    return await apiRequest('/api/compounds');
}

/**
 * 获取化合物详情
 */
async function getCompound(name) {
    return await apiRequest(`/api/compound/${encodeURIComponent(name)}`);
}

/**
 * 毒性预测
 */
async function predict(smiles = null, compound = null, model = 'stacking') {
    const data = { model: model };
    if (smiles) data.smiles = smiles;
    if (compound) data.compound = compound;
    return await apiRequest('/api/predict', 'POST', data);
}

/**
 * 对比分析
 */
async function compare(compound1, compound2) {
    return await apiRequest('/api/compare', 'POST', {
        compound1: compound1,
        compound2: compound2,
    });
}

/**
 * 智能问答
 */
async function askQuestion(question) {
    return await apiRequest('/api/qa', 'POST', { question: question });
}

/**
 * 生成报告
 */
async function generateReport(compound) {
    return await apiRequest('/api/report', 'POST', { compound: compound });
}

/**
 * 验证 SMILES
 */
async function validateSmiles(smiles) {
    return await apiRequest('/api/validate_smiles', 'POST', { smiles: smiles });
}

/**
 * 健康检查
 */
async function healthCheck() {
    return await apiRequest('/api/health');
}

/**
 * 格式化风险等级
 */
function formatRisk(level) {
    const colors = {
        '高': '#c62828',
        '中': '#e65100',
        '低': '#2e7d32',
        '高风险': '#c62828',
        '中风险': '#e65100',
        '低风险': '#2e7d32',
    };
    const color = colors[level] || '#666';
    return `<span style="color:${color};font-weight:bold">${level}</span>`;
}

/**
 * 格式化置信度
 */
function formatConfidence(confidence) {
    const percent = (confidence * 100).toFixed(0);
    return `${percent}%`;
}

/**
 * 复制到剪贴板
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        alert('已复制到剪贴板');
    }).catch(err => {
        console.error('复制失败:', err);
    });
}

/**
 * 下载为文件
 */
function downloadFile(content, filename, type = 'text/plain') {
    const blob = new Blob([content], { type: type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * 下载 Markdown 报告
 */
function downloadReport(content, compound) {
    const filename = `PFAS风险评估报告_${compound}_${new Date().toISOString().slice(0,10)}.md`;
    downloadFile(content, filename, 'text/markdown');
}

/**
 * 显示加载状态
 */
function showLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) {
        el.innerHTML = '<div class="loading">加载中...</div>';
    }
}

/**
 * 隐藏加载状态
 */
function hideLoading(elementId) {
    const el = document.getElementById(elementId);
    if (el) {
        el.innerHTML = '';
    }
}

/**
 * 显示错误消息
 */
function showError(message) {
    alert('错误: ' + message);
}

/**
 * 显示成功消息
 */
function showSuccess(message) {
    // 可以用更好的UI替换
    console.log('Success:', message);
}

// 页面加载完成后的初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('PFAS-Sentry 已加载');

    // 检查API健康状态
    healthCheck().then(result => {
        console.log('API状态:', result.status);
    }).catch(err => {
        console.warn('API连接失败:', err);
    });
});
