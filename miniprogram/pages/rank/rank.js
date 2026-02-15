const { getChampions } = require("../../services/api");
const { API_BASE } = require("../../config");

function toPercent(value) {
  if (typeof value !== "number") {
    return "-";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function toAbsoluteIcon(path, fallbackPath) {
  const raw = path || fallbackPath;
  if (!raw) {
    return "";
  }
  if (/^https?:\/\//.test(raw)) {
    return raw;
  }
  return `${API_BASE}${raw.startsWith("/") ? "" : "/"}${raw}`;
}

function normalizeRow(item, rank) {
  const championId = String(item.championId || "");
  const fullname = item.fullname_cn || item.name_cn || "未知英雄";
  return {
    championId,
    fullname_cn: fullname,
    winRate: item.winRate,
    pickRate: item.pickRate,
    winRateText: toPercent(item.winRate),
    pickRateText: toPercent(item.pickRate),
    iconUrl: toAbsoluteIcon(item.iconUrl, `/static/champions/${championId}.png`),
    rank
  };
}

Page({
  data: {
    qInput: "",
    q: "",
    sort: "winRate",
    order: "desc",
    limit: 20,
    offset: 0,
    list: [],
    loading: false,
    hasMore: true,
    error: null,
    avatarErrorMap: {}
  },

  onLoad() {
    this.renderFromAppCache();
    this.fetchChampions({ reset: true });
  },

  renderFromAppCache() {
    const app = getApp();
    const cached = app && app.globalData ? app.globalData.champions : [];
    if (!Array.isArray(cached) || cached.length === 0) {
      return;
    }

    const initial = cached
      .slice(0, this.data.limit)
      .map((item, idx) => normalizeRow(item, idx + 1));
    this.setData({
      list: initial,
      offset: initial.length,
      hasMore: cached.length > initial.length,
      error: null,
      avatarErrorMap: {}
    });
  },

  onSearchInput(e) {
    this.setData({
      qInput: (e.detail && e.detail.value) || ""
    });
  },

  onSearchConfirm() {
    this.setData({
      q: this.data.qInput.trim()
    });
    this.fetchChampions({ reset: true });
  },

  onTapSearch() {
    this.onSearchConfirm();
  },

  onToggleSort(e) {
    const nextSort = e.currentTarget.dataset.sort;
    if (!nextSort) {
      return;
    }

    let nextOrder = "desc";
    if (this.data.sort === nextSort) {
      nextOrder = this.data.order === "desc" ? "asc" : "desc";
    }

    this.setData({
      sort: nextSort,
      order: nextOrder
    });
    this.fetchChampions({ reset: true });
  },

  onToggleOrder() {
    this.setData({
      order: this.data.order === "desc" ? "asc" : "desc"
    });
    this.fetchChampions({ reset: true });
  },

  onLoadMore() {
    if (!this.data.hasMore || this.data.loading) {
      return;
    }
    this.fetchChampions({ reset: false });
  },

  onTapChampion(e) {
    const championId = e.currentTarget.dataset.id;
    const fullname = e.currentTarget.dataset.fullname || "";
    const iconUrl = e.currentTarget.dataset.icon || "";
    if (!championId) {
      return;
    }
    wx.navigateTo({
      url: `/pages/champion/champion?championId=${encodeURIComponent(
        championId
      )}&fullname=${encodeURIComponent(fullname)}&iconUrl=${encodeURIComponent(
        iconUrl
      )}`
    });
  },

  onHeroAvatarError(e) {
    const championId = String(e.currentTarget.dataset.id || "");
    if (!championId || this.data.avatarErrorMap[championId]) {
      return;
    }
    this.setData({
      [`avatarErrorMap.${championId}`]: true
    });
  },

  fetchChampions({ reset }) {
    if (this.data.loading) {
      return;
    }

    const nextOffset = reset ? 0 : this.data.offset;
    this.setData({
      loading: true,
      error: null
    });

    getChampions(
      {
        q: this.data.q,
        sort: this.data.sort,
        order: this.data.order,
        limit: this.data.limit,
        offset: nextOffset
      },
      {
        showLoading: false
      }
    )
      .then((rows) => {
        const normalizedBatch = Array.isArray(rows)
          ? rows.map((item, idx) => normalizeRow(item, nextOffset + idx + 1))
          : [];

        const nextList = reset
          ? normalizedBatch
          : this.data.list.concat(normalizedBatch);
        this.setData({
          list: nextList,
          offset: nextOffset + normalizedBatch.length,
          hasMore: normalizedBatch.length === this.data.limit,
          loading: false,
          error: null,
          avatarErrorMap: reset ? {} : this.data.avatarErrorMap
        });
      })
      .catch((err) => {
        const hasLocalData = this.data.list.length > 0;
        this.setData({
          loading: false,
          error: hasLocalData ? null : err.message || "加载失败"
        });
      });
  }
});
