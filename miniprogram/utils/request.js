const { API_BASE } = require("../config");

function buildUrl(path) {
  if (/^https?:\/\//.test(path)) {
    return path;
  }
  return `${API_BASE}${path}`;
}

function normalizeErrorMessage(data, statusCode) {
  if (statusCode === 503) {
    return "服务暂时不可用，请稍后重试";
  }
  if (data && typeof data === "object" && typeof data.detail === "string") {
    return data.detail;
  }
  return `请求失败 (${statusCode})`;
}

function request(options) {
  const {
    url,
    method = "GET",
    data = {},
    showLoading = true,
    loadingText = "加载中",
    showErrorToast = true,
    timeout = 12000
  } = options;

  if (!url) {
    return Promise.reject(new Error("request url is required"));
  }

  if (showLoading) {
    wx.showLoading({
      title: loadingText,
      mask: true
    });
  }

  return new Promise((resolve, reject) => {
    wx.request({
      url: buildUrl(url),
      method,
      data,
      timeout,
      success: (res) => {
        const { statusCode, data: resData } = res;
        if (statusCode >= 200 && statusCode < 300) {
          resolve(resData);
          return;
        }

        const message = normalizeErrorMessage(resData, statusCode);
        if (showErrorToast) {
          wx.showToast({
            title: message,
            icon: "none",
            duration: 2200
          });
        }
        reject(new Error(message));
      },
      fail: (err) => {
        const message =
          err && err.errMsg ? err.errMsg : "网络请求失败，请检查网络连接";
        if (showErrorToast) {
          wx.showToast({
            title: message,
            icon: "none",
            duration: 2200
          });
        }
        reject(new Error(message));
      },
      complete: () => {
        if (showLoading) {
          wx.hideLoading();
        }
      }
    });
  });
}

module.exports = {
  request
};
