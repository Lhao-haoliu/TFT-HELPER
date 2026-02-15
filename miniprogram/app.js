const { getAugmentMapping, getChampions } = require("./services/api");

const CACHE_CHAMPIONS = "CACHE_CHAMPIONS";
const CACHE_AUGMENT_MAPPING = "CACHE_AUGMENT_MAPPING";
const CACHE_UPDATED_AT = "CACHE_UPDATED_AT";

function toDayString(timestamp) {
  const d = new Date(timestamp);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function isToday(timestamp) {
  if (!timestamp) {
    return false;
  }
  return toDayString(timestamp) === toDayString(Date.now());
}

App({
  globalData: {
    champions: [],
    augmentMapping: {},
    lastUpdated: null
  },

  onLaunch() {
    this.loadCacheToGlobalData();
    this.refreshGlobalCacheInBackground();
  },

  loadCacheToGlobalData() {
    try {
      const champions = wx.getStorageSync(CACHE_CHAMPIONS);
      const augmentMapping = wx.getStorageSync(CACHE_AUGMENT_MAPPING);
      const updatedAt = wx.getStorageSync(CACHE_UPDATED_AT);
      this.globalData.champions = Array.isArray(champions) ? champions : [];
      this.globalData.augmentMapping =
        augmentMapping && typeof augmentMapping === "object" ? augmentMapping : {};
      this.globalData.lastUpdated = typeof updatedAt === "number" ? updatedAt : null;
    } catch (err) {
      this.globalData.champions = [];
      this.globalData.augmentMapping = {};
      this.globalData.lastUpdated = null;
    }
  },

  refreshGlobalCacheInBackground() {
    const forceRefresh = !isToday(this.globalData.lastUpdated);
    this.fetchAndUpdateCache({
      forceRefresh
    });
  },

  fetchAndUpdateCache({ forceRefresh }) {
    const requestOptions = {
      showLoading: false,
      showErrorToast: false
    };

    const championsPromise = getChampions(
      {
        sort: "winRate",
        order: "desc",
        limit: 200,
        offset: 0
      },
      requestOptions
    );
    const mappingPromise = getAugmentMapping(requestOptions);

    Promise.all([championsPromise, mappingPromise])
      .then(([champions, augmentMapping]) => {
        const nextChampions = Array.isArray(champions) ? champions : [];
        const nextMapping =
          augmentMapping && typeof augmentMapping === "object" ? augmentMapping : {};
        const now = Date.now();
        this.globalData.champions = nextChampions;
        this.globalData.augmentMapping = nextMapping;
        this.globalData.lastUpdated = now;
        wx.setStorageSync(CACHE_CHAMPIONS, nextChampions);
        wx.setStorageSync(CACHE_AUGMENT_MAPPING, nextMapping);
        wx.setStorageSync(CACHE_UPDATED_AT, now);
      })
      .catch(() => {
        if (forceRefresh) {
          // Force-refresh mode still keeps old cache when network fails.
        }
      });
  }
});
