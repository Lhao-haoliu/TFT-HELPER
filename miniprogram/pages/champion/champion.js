const {
  getChampionAugments,
  getChampionCombos,
  getAugmentMapping
} = require("../../services/api");
const { API_BASE } = require("../../config");

const RARITY_TABS = [
  { label: "全部", value: "all" },
  { label: "棱彩", value: "prismatic" },
  { label: "黄金", value: "gold" },
  { label: "白银", value: "silver" }
];

const SORT_MODE_TABS = [
  { label: "胜率优先", value: "winRate" },
  { label: "综合评估", value: "comprehensive" }
];

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

function normalizeRarity(value) {
  if (!value || typeof value !== "string") {
    return "unknown";
  }
  const text = value.trim().toLowerCase();
  if (!text) {
    return "unknown";
  }
  if (text.includes("prismatic") || text.includes("kprismatic") || text.includes("棱彩")) {
    return "prismatic";
  }
  if (text.includes("gold") || text.includes("kgold") || text.includes("黄金")) {
    return "gold";
  }
  if (text.includes("silver") || text.includes("ksilver") || text.includes("白银")) {
    return "silver";
  }
  return "unknown";
}

function getTierRank(tier) {
  if (typeof tier !== "string") {
    return 999;
  }
  const matched = tier.toUpperCase().match(/T\s*(\d+)/);
  if (!matched) {
    return 999;
  }
  const value = Number(matched[1]);
  if (!Number.isFinite(value)) {
    return 999;
  }
  return value;
}

function normalizeAugment(raw, augmentMapping) {
  const augmentId = Number(raw && raw.augmentId);
  if (!Number.isFinite(augmentId)) {
    return null;
  }

  const mapped =
    augmentMapping && typeof augmentMapping === "object"
      ? augmentMapping[String(augmentId)] || {}
      : {};

  const tierRaw =
    typeof raw.tier === "string" && raw.tier.trim()
      ? raw.tier.trim()
      : typeof mapped.tier === "string" && mapped.tier.trim()
      ? mapped.tier.trim()
      : "-";

  const rarity = normalizeRarity(raw.rarity || mapped.rarity);

  return {
    augmentId,
    name_cn: raw.name_cn || mapped.name_cn || `#${augmentId}`,
    tier: tierRaw.toUpperCase(),
    rarity,
    winRate: typeof raw.winRate === "number" ? raw.winRate : 0,
    pickRate: typeof raw.pickRate === "number" ? raw.pickRate : 0,
    winRateText: toPercent(raw.winRate),
    pickRateText: toPercent(raw.pickRate),
    gamesText:
      typeof raw.games === "number" && raw.games >= 0 ? `${raw.games}` : "-",
    iconUrl: toAbsoluteIcon(
      raw.iconUrl || mapped.icon,
      `/static/augments/${augmentId}.png`
    )
  };
}

function applyAugmentView(augmentList, activeRarityTab, sortMode) {
  const source = Array.isArray(augmentList) ? augmentList : [];
  const filtered =
    activeRarityTab === "all"
      ? source.slice()
      : source.filter((item) => item.rarity === activeRarityTab);

  filtered.sort((a, b) => {
    if (sortMode === "comprehensive") {
      const tierDiff = getTierRank(a.tier) - getTierRank(b.tier);
      if (tierDiff !== 0) {
        return tierDiff;
      }
      const winDiff = b.winRate - a.winRate;
      if (winDiff !== 0) {
        return winDiff;
      }
      const pickDiff = b.pickRate - a.pickRate;
      if (pickDiff !== 0) {
        return pickDiff;
      }
      return a.augmentId - b.augmentId;
    }

    const winDiff = b.winRate - a.winRate;
    if (winDiff !== 0) {
      return winDiff;
    }
    const pickDiff = b.pickRate - a.pickRate;
    if (pickDiff !== 0) {
      return pickDiff;
    }
    const tierDiff = getTierRank(a.tier) - getTierRank(b.tier);
    if (tierDiff !== 0) {
      return tierDiff;
    }
    return a.augmentId - b.augmentId;
  });

  return filtered;
}

Page({
  data: {
    championId: "",
    fullname_cn: "",
    heroIconUrl: "",
    heroAvatarError: false,
    version: null,

    rarityTabs: RARITY_TABS,
    activeRarityTab: "all",
    sortModeTabs: SORT_MODE_TABS,
    sortMode: "winRate",

    augmentList: [],
    visibleAugmentList: [],
    augmentLoading: false,
    augmentMapping: {},
    augmentIconErrorMap: {},

    combosLoading: false,
    combos: [],
    errorAugments: null,
    errorCombos: null
  },

  onLoad(options) {
    const championId = options && options.championId ? `${options.championId}` : "";
    const fullname = options && options.fullname ? decodeURIComponent(options.fullname) : "";
    const iconFromQuery = options && options.iconUrl ? decodeURIComponent(options.iconUrl) : "";

    this.setData({
      championId,
      fullname_cn: fullname || "英雄详情",
      heroIconUrl: toAbsoluteIcon(iconFromQuery, `/static/champions/${championId}.png`),
      heroAvatarError: false
    });

    if (!championId) {
      wx.showToast({
        title: "缺少 championId",
        icon: "none"
      });
      return;
    }

    this.loadAugmentMapping();
    this.fetchAugments();
    this.fetchCombos();
  },

  onHeroAvatarError() {
    if (this.data.heroAvatarError) {
      return;
    }
    this.setData({
      heroAvatarError: true
    });
  },

  onAugmentIconError(e) {
    const id = String(e.currentTarget.dataset.id || "");
    if (!id || this.data.augmentIconErrorMap[id]) {
      return;
    }
    this.setData({
      [`augmentIconErrorMap.${id}`]: true
    });
  },

  onChangeRarityTab(e) {
    const nextTab = e.currentTarget.dataset.value;
    if (!nextTab || nextTab === this.data.activeRarityTab) {
      return;
    }
    this.setData(
      {
        activeRarityTab: nextTab
      },
      () => {
        this.refreshVisibleAugments();
      }
    );
  },

  onChangeSortMode(e) {
    const nextMode = e.currentTarget.dataset.value;
    if (!nextMode || nextMode === this.data.sortMode) {
      return;
    }
    this.setData(
      {
        sortMode: nextMode
      },
      () => {
        this.refreshVisibleAugments();
      }
    );
  },

  loadAugmentMapping() {
    getAugmentMapping({ showLoading: false })
      .then((mapping) => {
        if (!mapping || typeof mapping !== "object") {
          return;
        }

        const rebuilt = (this.data.augmentList || [])
          .map((item) => normalizeAugment(item, mapping))
          .filter((item) => !!item);

        this.setData(
          {
            augmentMapping: mapping,
            augmentList: rebuilt
          },
          () => {
            this.refreshVisibleAugments();
          }
        );
      })
      .catch(() => {
        // Mapping is optional for rarity fallback.
      });
  },

  refreshVisibleAugments() {
    const visibleAugmentList = applyAugmentView(
      this.data.augmentList,
      this.data.activeRarityTab,
      this.data.sortMode
    );
    this.setData({
      visibleAugmentList
    });
  },

  fetchAugments() {
    if (this.data.augmentLoading) {
      return;
    }

    this.setData({
      augmentLoading: true,
      errorAugments: null
    });

    getChampionAugments(
      this.data.championId,
      {
        sort: "winRate",
        order: "desc"
      },
      {
        showLoading: false
      }
    )
      .then((rows) => {
        const mapping = this.data.augmentMapping || {};
        const normalized = Array.isArray(rows)
          ? rows.map((row) => normalizeAugment(row, mapping)).filter((item) => !!item)
          : [];

        this.setData(
          {
            augmentList: normalized,
            augmentLoading: false,
            errorAugments: null,
            augmentIconErrorMap: {}
          },
          () => {
            this.refreshVisibleAugments();
          }
        );
      })
      .catch((err) => {
        this.setData({
          augmentLoading: false,
          errorAugments: err.message || "海克斯推荐加载失败"
        });
      });
  },

  fetchCombos() {
    this.setData({
      combosLoading: true,
      errorCombos: null
    });

    getChampionCombos(
      this.data.championId,
      {},
      {
        showLoading: false
      }
    )
      .then((res) => {
        const combos = Array.isArray(res.combos) ? res.combos.slice(0, 10) : [];
        const normalized = combos.map((combo) => {
          const augments = Array.isArray(combo.augments) ? combo.augments : [];
          return {
            rank: combo.rank,
            augments
          };
        });

        this.setData({
          version: res.version || null,
          combos: normalized,
          combosLoading: false
        });
      })
      .catch((err) => {
        this.setData({
          combosLoading: false,
          errorCombos: err.message || "海克斯组合加载失败"
        });
      });
  }
});
