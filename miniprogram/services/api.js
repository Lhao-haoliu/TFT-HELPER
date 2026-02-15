const { request } = require("../utils/request");
const { API_BASE } = require("../config");

function uploadImageForOcr(imagePath, options = {}) {
  const { showLoading = true, loadingText = "Recognizing...", showErrorToast = true } = options;

  if (!imagePath) {
    return Promise.reject(new Error("imagePath is required"));
  }

  if (showLoading) {
    wx.showLoading({
      title: loadingText,
      mask: true
    });
  }

  return new Promise((resolve, reject) => {
    wx.uploadFile({
      url: `${API_BASE}/api/ocr/augments-names`,
      filePath: imagePath,
      name: "image",
      success: (res) => {
        const statusCode = res.statusCode;
        let data = {};
        try {
          data = JSON.parse(res.data || "{}");
        } catch (err) {
          data = {};
        }

        if (statusCode >= 200 && statusCode < 300) {
          resolve(data);
          return;
        }

        const message =
          (data &&
            ((data.error && data.error.message) ||
              (typeof data.detail === "string" ? data.detail : ""))) ||
          `OCR request failed (${statusCode})`;
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
        const message = (err && err.errMsg) || "OCR upload failed. Check network.";
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

function health(requestOptions = {}) {
  return request({
    url: "/health",
    method: "GET",
    showLoading: false,
    ...requestOptions
  });
}

function getChampions(params = {}, requestOptions = {}) {
  return request({
    url: "/api/champions",
    method: "GET",
    data: params,
    ...requestOptions
  });
}

function getChampionAugments(championId, params = {}, requestOptions = {}) {
  return request({
    url: `/api/champions/${championId}/augments`,
    method: "GET",
    data: params,
    ...requestOptions
  });
}

function getChampionCombos(championId, params = {}, requestOptions = {}) {
  return request({
    url: `/api/champions/${championId}/augment-combos`,
    method: "GET",
    data: params,
    ...requestOptions
  });
}

function getAugmentMapping(requestOptions = {}) {
  return request({
    url: "/api/augments/mapping",
    method: "GET",
    ...requestOptions
  });
}

function resolveAugmentNames(names = [], requestOptions = {}) {
  return request({
    url: "/api/augments/resolve-names",
    method: "POST",
    data: {
      names
    },
    ...requestOptions
  });
}

function battleRecommend(payload = {}, requestOptions = {}) {
  return request({
    url: "/api/battle/recommend",
    method: "POST",
    data: payload,
    ...requestOptions
  });
}

module.exports = {
  health,
  getChampions,
  getChampionAugments,
  getChampionCombos,
  getAugmentMapping,
  uploadImageForOcr,
  resolveAugmentNames,
  battleRecommend
};
