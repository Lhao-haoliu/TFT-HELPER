const {
  getChampions,
  getAugmentMapping,
  uploadImageForOcr,
  resolveAugmentNames,
  battleRecommend
} = require("../../services/api");
const { API_BASE } = require("../../config");

const MAX_ROUND = 4;

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

function normalizeHero(item) {
  const championId = String(item.championId || "");
  const fullname = item.fullname_cn || item.name_cn || `英雄 ${championId}`;
  return {
    championId,
    fullname_cn: fullname,
    iconUrl: toAbsoluteIcon(item.iconUrl, `/static/champions/${championId}.png`),
    winRate: typeof item.winRate === "number" ? item.winRate : 0,
    winRateText: toPercent(item.winRate)
  };
}

function buildSummaryText(champion, pickedAugments) {
  const heroName = champion && champion.fullname_cn ? champion.fullname_cn : "未选择英雄";
  const lines = [`英雄：${heroName}`, "已选海克斯："];
  if (!Array.isArray(pickedAugments) || pickedAugments.length === 0) {
    lines.push("- 无");
  } else {
    pickedAugments.forEach((item, idx) => {
      lines.push(`${idx + 1}. ${item.name_cn}`);
    });
  }
  return lines.join("\n");
}

Page({
  data: {
    state: {
      step: "pickHero",
      champion: null,
      round: 1,
      candidates: [],
      picks: [],
      recommendSingle: null,
      recommendSingleName: "",
      recommendCombo: null,
      recommendComboName: "",
      comboExplain: ""
    },

    pageLoading: false,
    pageError: null,

    heroQuery: "",
    heroList: [],
    filteredHeroes: [],
    heroAvatarErrorMap: {},

    augmentMapping: {},
    augmentSearchList: [],

    pickerVisible: false,
    pickerAugmentIdInput: "",
    pickerSearchInput: "",
    pickerSearchResults: [],

    roundImages: {},
    currentRoundImage: "",
    currentHeroAvatarError: false,

    iconErrorMap: {},
    pickedAugments: [],
    summaryText: "",

    ocrLoading: false,
    ocrError: null
  },

  onLoad() {
    this.bootstrapPage();
  },

  setBattleState(patch, callback) {
    const nextState = {
      ...(this.data.state || {}),
      ...patch
    };
    this.setData(
      {
        state: nextState
      },
      callback
    );
  },

  bootstrapPage() {
    this.setData({
      pageLoading: true,
      pageError: null
    });

    const app = getApp();
    const cacheHeroes = app && app.globalData ? app.globalData.champions : [];
    const cacheMapping = app && app.globalData ? app.globalData.augmentMapping : {};

    let heroesPromise;
    if (Array.isArray(cacheHeroes) && cacheHeroes.length > 0) {
      heroesPromise = Promise.resolve(cacheHeroes);
    } else {
      heroesPromise = getChampions(
        {
          sort: "winRate",
          order: "desc",
          limit: 200,
          offset: 0
        },
        {
          showLoading: false
        }
      );
    }

    let mappingPromise;
    if (cacheMapping && typeof cacheMapping === "object" && Object.keys(cacheMapping).length > 0) {
      mappingPromise = Promise.resolve(cacheMapping);
    } else {
      mappingPromise = getAugmentMapping({
        showLoading: false
      });
    }

    Promise.all([heroesPromise, mappingPromise])
      .then(([heroes, mapping]) => {
        const heroList = Array.isArray(heroes) ? heroes.map(normalizeHero) : [];
        const safeMapping = mapping && typeof mapping === "object" ? mapping : {};
        const augmentSearchList = this.buildAugmentSearchList(safeMapping);

        this.setData({
          heroList,
          filteredHeroes: heroList,
          augmentMapping: safeMapping,
          augmentSearchList,
          pageLoading: false,
          pageError: null
        });
      })
      .catch((err) => {
        this.setData({
          pageLoading: false,
          pageError: err.message || "加载失败"
        });
      });
  },

  buildAugmentSearchList(mapping) {
    const list = [];
    Object.keys(mapping || {}).forEach((key) => {
      const raw = mapping[key] || {};
      const augmentId = Number(raw.augmentId || key);
      if (!Number.isFinite(augmentId)) {
        return;
      }
      list.push({
        augmentId,
        name_cn: raw.name_cn || `#${augmentId}`,
        rarity: normalizeRarity(raw.rarity),
        iconUrl: toAbsoluteIcon(raw.icon, `/static/augments/${augmentId}.png`)
      });
    });
    list.sort((a, b) => {
      if (a.name_cn === b.name_cn) {
        return a.augmentId - b.augmentId;
      }
      return a.name_cn.localeCompare(b.name_cn, "zh-Hans-CN");
    });
    return list;
  },

  onHeroSearchInput(e) {
    const heroQuery = (e.detail && e.detail.value) || "";
    const normalized = heroQuery.trim().toLowerCase();
    const filteredHeroes = normalized
      ? this.data.heroList.filter((item) => item.fullname_cn.toLowerCase().includes(normalized))
      : this.data.heroList;

    this.setData({
      heroQuery,
      filteredHeroes
    });
  },

  onHeroAvatarError(e) {
    const cid = String(e.currentTarget.dataset.id || "");
    if (!cid || this.data.heroAvatarErrorMap[cid]) {
      return;
    }
    this.setData({
      [`heroAvatarErrorMap.${cid}`]: true
    });
  },

  onPickHero(e) {
    const championId = String(e.currentTarget.dataset.id || "");
    if (!championId) {
      return;
    }
    const hero = (this.data.heroList || []).find((item) => item.championId === championId);
    if (!hero) {
      return;
    }

    this.startRoundFlow(hero);
  },

  startRoundFlow(hero) {
    this.setData({
      pageLoading: false,
      pageError: null,
      iconErrorMap: {},
      roundImages: {},
      currentRoundImage: "",
      currentHeroAvatarError: false,
      pickedAugments: [],
      summaryText: "",
      ocrLoading: false,
      ocrError: null
    });

    this.setBattleState({
      step: "round",
      champion: {
        championId: hero.championId,
        fullname_cn: hero.fullname_cn,
        iconUrl: hero.iconUrl
      },
      round: 1,
      candidates: [],
      picks: [],
      recommendSingle: null,
      recommendSingleName: "",
      recommendCombo: null,
      recommendComboName: "",
      comboExplain: ""
    });
  },

  getCandidateFromId(augmentId) {
    const id = String(augmentId);
    const mapped = this.data.augmentMapping[id] || {};
    const winRate = null;
    const pickRate = null;

    return {
      augmentId: Number(augmentId),
      name_cn: mapped.name_cn || `#${augmentId}`,
      iconUrl: toAbsoluteIcon(mapped.icon, `/static/augments/${augmentId}.png`),
      winRate,
      pickRate,
      tier: "-",
      rarity: normalizeRarity(mapped.rarity),
      winRateText: toPercent(winRate),
      pickRateText: toPercent(pickRate)
    };
  },

  normalizeCandidateFromRecommend(item) {
    const augmentId = Number(item && item.augmentId);
    if (!Number.isFinite(augmentId)) {
      return null;
    }
    const winRate = typeof item.winRate === "number" ? item.winRate : null;
    const pickRate = typeof item.pickRate === "number" ? item.pickRate : null;
    const tier = typeof item.tier === "string" && item.tier.trim() ? item.tier.trim() : "-";
    const rarity = normalizeRarity(item.rarity);
    return {
      augmentId,
      name_cn: item.name_cn || `#${augmentId}`,
      iconUrl: toAbsoluteIcon(item.iconUrl, `/static/augments/${augmentId}.png`),
      winRate,
      pickRate,
      tier: tier.toUpperCase(),
      rarity,
      winRateText: toPercent(winRate),
      pickRateText: toPercent(pickRate)
    };
  },

  async requestBattleRecommendation() {
    const state = this.data.state || {};
    const champion = state.champion || null;
    const candidates = state.candidates || [];
    if (!champion || !champion.championId || candidates.length === 0) {
      this.setBattleState({
        recommendSingle: null,
        recommendSingleName: "",
        recommendCombo: null,
        recommendComboName: "",
        comboExplain: ""
      });
      return;
    }

    const payload = {
      championId: Number(champion.championId) || champion.championId,
      round: Number(state.round || 1),
      pickedAugmentIds: (state.picks || []).map((id) => Number(id)).filter((id) => Number.isFinite(id)),
      candidateAugmentIds: candidates
        .map((item) => Number(item.augmentId))
        .filter((id) => Number.isFinite(id))
    };

    try {
      const resp = await battleRecommend(payload, {
        showLoading: false,
        showErrorToast: false
      });
      const normalizedCandidates = Array.isArray(resp && resp.candidates)
        ? resp.candidates.map((row) => this.normalizeCandidateFromRecommend(row)).filter((row) => !!row)
        : [];

      const recommendSingle = resp && resp.recommendSingle ? resp.recommendSingle : null;
      const recommendCombo = resp && resp.recommendCombo ? resp.recommendCombo : null;

      this.setBattleState({
        candidates: normalizedCandidates,
        recommendSingle:
          recommendSingle && Number.isFinite(Number(recommendSingle.augmentId))
            ? Number(recommendSingle.augmentId)
            : null,
        recommendSingleName:
          recommendSingle && typeof recommendSingle.name_cn === "string"
            ? recommendSingle.name_cn
            : "",
        recommendCombo:
          recommendCombo && Number.isFinite(Number(recommendCombo.augmentId))
            ? Number(recommendCombo.augmentId)
            : null,
        recommendComboName:
          recommendCombo && typeof recommendCombo.name_cn === "string"
            ? recommendCombo.name_cn
            : "",
        comboExplain: (() => {
          if (!recommendCombo || typeof recommendCombo.reason !== "string") {
            return "";
          }
          const rank = Number(recommendCombo.comboRank);
          if (Number.isFinite(rank) && rank > 0) {
            return `${recommendCombo.reason}（第${rank}组）`;
          }
          return recommendCombo.reason;
        })()
      });
    } catch (err) {
      // Keep current candidate list for manual continuation.
    }
  },

  normalizeResolvedName(name) {
    return String(name || "").replace(/\s+/g, "").trim();
  },

  fallbackResolveNamesLocally(names) {
    const mapping = this.data.augmentMapping || {};
    const normalizedMap = {};
    Object.keys(mapping).forEach((key) => {
      const item = mapping[key] || {};
      const name = this.normalizeResolvedName(item.name_cn || "");
      if (!name) {
        return;
      }
      if (!(name in normalizedMap)) {
        normalizedMap[name] = Number(item.augmentId || key);
      }
    });

    return names.map((name) => {
      const normalized = this.normalizeResolvedName(name);
      const augmentId = normalizedMap[normalized];
      if (Number.isFinite(augmentId)) {
        return {
          name_cn: name,
          augmentId,
          confidence: 0.75
        };
      }
      return {
        name_cn: name,
        augmentId: null,
        confidence: 0
      };
    });
  },

  async autoDetectCandidatesFromImage(imagePath) {
    this.setData({
      ocrLoading: true,
      ocrError: null
    });

    try {
      const ocrResp = await uploadImageForOcr(imagePath, {
        showLoading: false,
        showErrorToast: false
      });

      const names = Array.isArray(ocrResp && ocrResp.names)
        ? ocrResp.names.map((item) => (item && item.name_cn ? String(item.name_cn).trim() : ""))
        : [];
      const validNames = names.filter((name) => !!name);
      if (validNames.length === 0) {
        throw new Error("未识别到海克斯名称");
      }

      let resolvedRows = [];
      try {
        const resolvedResp = await resolveAugmentNames(validNames, {
          showLoading: false,
          showErrorToast: false
        });
        resolvedRows = Array.isArray(resolvedResp && resolvedResp.resolved)
          ? resolvedResp.resolved
          : [];
      } catch (err) {
        resolvedRows = this.fallbackResolveNamesLocally(validNames);
      }

      const picksSet = new Set((this.data.state.picks || []).map((id) => Number(id)));
      const idSet = new Set();
      const candidates = [];
      resolvedRows.forEach((row) => {
        const augmentId = Number(row && row.augmentId);
        if (!Number.isFinite(augmentId)) {
          return;
        }
        if (picksSet.has(augmentId) || idSet.has(augmentId)) {
          return;
        }
        idSet.add(augmentId);
        candidates.push(this.getCandidateFromId(augmentId));
      });

      if (candidates.length === 0) {
        throw new Error("识别结果无法映射为海克斯ID");
      }

      const finalCandidates = candidates.slice(0, 3);
      this.setBattleState(
        {
          candidates: finalCandidates,
          recommendSingle: null,
          recommendSingleName: "",
          recommendCombo: null,
          recommendComboName: "",
          comboExplain: ""
        },
        () => this.requestBattleRecommendation()
      );

      this.setData({
        ocrLoading: false,
        ocrError:
          finalCandidates.length < 3
            ? "自动识别不足3个候选，请手动补全"
            : null,
        iconErrorMap: {}
      });
    } catch (err) {
      this.setData({
        ocrLoading: false,
        ocrError: (err && err.message) || "识别失败，请手动选择候选"
      });
    }
  },

  openCandidatePicker() {
    this.setData({
      pickerVisible: true,
      pickerAugmentIdInput: "",
      pickerSearchInput: "",
      pickerSearchResults: this.data.augmentSearchList.slice(0, 30)
    });
  },

  closeCandidatePicker() {
    this.setData({
      pickerVisible: false,
      pickerAugmentIdInput: "",
      pickerSearchInput: "",
      pickerSearchResults: []
    });
  },

  onCandidateIdInput(e) {
    this.setData({
      pickerAugmentIdInput: (e.detail && e.detail.value) || ""
    });
  },

  onAddCandidateById() {
    const input = (this.data.pickerAugmentIdInput || "").trim();
    if (!input || !/^\d+$/.test(input)) {
      wx.showToast({ title: "请输入有效 augmentId", icon: "none" });
      return;
    }
    this.tryAddCandidate(Number(input));
  },

  onCandidateSearchInput(e) {
    const keyword = ((e.detail && e.detail.value) || "").trim().toLowerCase();
    const results = keyword
      ? this.data.augmentSearchList
          .filter((item) => item.name_cn.toLowerCase().includes(keyword))
          .slice(0, 40)
      : this.data.augmentSearchList.slice(0, 30);

    this.setData({
      pickerSearchInput: (e.detail && e.detail.value) || "",
      pickerSearchResults: results
    });
  },

  onPickSearchResult(e) {
    const augmentId = Number(e.currentTarget.dataset.id);
    if (!Number.isFinite(augmentId)) {
      return;
    }
    this.tryAddCandidate(augmentId);
  },

  tryAddCandidate(augmentId) {
    const state = this.data.state || {};
    const candidates = state.candidates || [];

    if (candidates.length >= 3) {
      wx.showToast({ title: "本轮最多3个候选", icon: "none" });
      return;
    }

    if ((state.picks || []).includes(augmentId)) {
      wx.showToast({ title: "该海克斯已选过", icon: "none" });
      return;
    }

    if (candidates.some((item) => item.augmentId === augmentId)) {
      wx.showToast({ title: "候选中已存在", icon: "none" });
      return;
    }

    const candidate = this.getCandidateFromId(augmentId);
    if (!candidate.name_cn) {
      wx.showToast({ title: "未找到该海克斯", icon: "none" });
      return;
    }

    const nextCandidates = candidates.concat(candidate);
    this.setBattleState(
      {
        candidates: nextCandidates
      },
      () => this.requestBattleRecommendation()
    );

    this.setData({
      pickerAugmentIdInput: "",
      pickerSearchInput: "",
      pickerSearchResults: this.data.augmentSearchList.slice(0, 30)
    });

    if (nextCandidates.length >= 3) {
      this.closeCandidatePicker();
    }
  },

  onRemoveCandidate(e) {
    const augmentId = Number(e.currentTarget.dataset.id);
    if (!Number.isFinite(augmentId)) {
      return;
    }

    const candidates = (this.data.state.candidates || []).filter(
      (item) => item.augmentId !== augmentId
    );

    this.setBattleState(
      {
        candidates
      },
      () => this.requestBattleRecommendation()
    );
  },

  getPickedDisplayList(picks) {
    return (picks || []).map((id) => {
      const detail = this.getCandidateFromId(id);
      return {
        augmentId: Number(id),
        name_cn: detail.name_cn || `#${id}`
      };
    });
  },

  onPickCandidate(e) {
    const augmentId = Number(e.currentTarget.dataset.id);
    if (!Number.isFinite(augmentId)) {
      return;
    }

    const state = this.data.state || {};
    const candidates = state.candidates || [];
    const selected = candidates.find((item) => item.augmentId === augmentId);
    if (!selected) {
      return;
    }

    if (candidates.length < 3) {
      wx.showToast({ title: "请先凑齐3个候选", icon: "none" });
      return;
    }

    wx.showModal({
      title: "确认选择",
      content: `本轮选择「${selected.name_cn}」？`,
      confirmText: "确认",
      cancelText: "取消",
      success: (res) => {
        if (!res.confirm) {
          return;
        }
        this.confirmRoundPick(selected.augmentId);
      }
    });
  },

  confirmRoundPick(augmentId) {
    const state = this.data.state || {};
    const round = Number(state.round || 1);
    const nextPicks = (state.picks || []).concat(augmentId);

    if (round >= MAX_ROUND || nextPicks.length >= MAX_ROUND) {
      this.setBattleState(
        {
          picks: nextPicks,
          step: "done",
          candidates: [],
          recommendSingle: null,
          recommendSingleName: "",
          recommendCombo: null,
          recommendComboName: "",
          comboExplain: ""
        },
        () => {
          this.buildSummary();
        }
      );
      return;
    }

    const nextRound = round + 1;
    const nextImage = this.data.roundImages[String(nextRound)] || "";
    const pickedDisplay = this.getPickedDisplayList(nextPicks);

    this.setBattleState({
      picks: nextPicks,
      round: nextRound,
      candidates: [],
      recommendSingle: null,
      recommendSingleName: "",
      recommendCombo: null,
      recommendComboName: "",
      comboExplain: ""
    });

    this.setData({
      currentRoundImage: nextImage,
      iconErrorMap: {},
      ocrError: null,
      pickedAugments: pickedDisplay
    });

    wx.showToast({
      title: `已进入第${nextRound}轮`,
      icon: "none",
      duration: 1600
    });
    wx.pageScrollTo({
      scrollTop: 0,
      duration: 220
    });
  },

  onResetCurrentRound() {
    this.setBattleState({
      candidates: [],
      recommendSingle: null,
      recommendSingleName: "",
      recommendCombo: null,
      recommendComboName: "",
      comboExplain: ""
    });
    this.setData({
      iconErrorMap: {},
      ocrError: null
    });
  },

  onEndBattleNow() {
    wx.showModal({
      title: "结束实战分析",
      content: "确认结束并查看总结？",
      success: (res) => {
        if (!res.confirm) {
          return;
        }
        this.setBattleState(
          {
            step: "done",
            candidates: [],
            recommendSingle: null,
            recommendSingleName: "",
            recommendCombo: null,
            recommendComboName: "",
            comboExplain: ""
          },
          () => {
            this.buildSummary();
          }
        );
      }
    });
  },

  onChooseRoundImage() {
    const round = String(this.data.state.round || 1);
    wx.chooseMedia({
      count: 1,
      mediaType: ["image"],
      sourceType: ["camera", "album"],
      success: (res) => {
        const file = Array.isArray(res.tempFiles) && res.tempFiles[0] ? res.tempFiles[0] : null;
        const path = file && file.tempFilePath ? file.tempFilePath : "";
        if (!path) {
          return;
        }
        this.setBattleState({
          candidates: [],
          recommendSingle: null,
          recommendSingleName: "",
          recommendCombo: null,
          recommendComboName: "",
          comboExplain: ""
        });
        this.setData({
          [`roundImages.${round}`]: path,
          currentRoundImage: path,
          ocrError: null,
          iconErrorMap: {}
        });
        this.autoDetectCandidatesFromImage(path);
      }
    });
  },

  onCurrentHeroAvatarError() {
    if (this.data.currentHeroAvatarError) {
      return;
    }
    this.setData({
      currentHeroAvatarError: true
    });
  },

  onPreviewRoundImage() {
    const current = this.data.currentRoundImage;
    if (!current) {
      return;
    }
    wx.previewImage({
      current,
      urls: [current]
    });
  },

  onCandidateIconError(e) {
    const key = String(e.currentTarget.dataset.id || "");
    if (!key || this.data.iconErrorMap[key]) {
      return;
    }
    this.setData({
      [`iconErrorMap.${key}`]: true
    });
  },

  buildSummary() {
    const state = this.data.state || {};
    const picks = state.picks || [];
    const details = picks.map((id) => this.getCandidateFromId(id));
    const summaryText = buildSummaryText(state.champion, details);

    this.setData({ pickedAugments: details, summaryText });
  },

  onCopySummary() {
    const text = this.data.summaryText || "";
    if (!text) {
      return;
    }
    wx.setClipboardData({
      data: text,
      success: () => {
        wx.showToast({
          title: "已复制",
          icon: "none"
        });
      }
    });
  },

  onRestart() {
    this.setData({
      pageError: null,
      iconErrorMap: {},
      pickedAugments: [],
      summaryText: "",
      roundImages: {},
      currentRoundImage: "",
      currentHeroAvatarError: false,
      pickerVisible: false,
      pickerAugmentIdInput: "",
      pickerSearchInput: "",
      pickerSearchResults: [],
      heroAvatarErrorMap: {},
      ocrLoading: false,
      ocrError: null
    });

    this.setBattleState({
      step: "pickHero",
      champion: null,
      round: 1,
      candidates: [],
      picks: [],
      recommendSingle: null,
      recommendSingleName: "",
      recommendCombo: null,
      recommendComboName: "",
      comboExplain: ""
    });
  }
});
