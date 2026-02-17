const DEV_API_BASE = "http://127.0.0.1:8000";
const PROD_API_BASE = "https://hksdld-tool-226173-8-1391225113.sh.run.tcloudbase.com";
const FORCE_PROD_API = true;

function resolveApiBase() {
  if (FORCE_PROD_API) {
    return PROD_API_BASE;
  }

  try {
    const account =
      typeof wx !== "undefined" && wx.getAccountInfoSync
        ? wx.getAccountInfoSync()
        : null;
    const envVersion =
      account &&
      account.miniProgram &&
      typeof account.miniProgram.envVersion === "string"
        ? account.miniProgram.envVersion
        : "develop";

    return envVersion === "develop" ? DEV_API_BASE : PROD_API_BASE;
  } catch (err) {
    return DEV_API_BASE;
  }
}

const API_BASE = resolveApiBase();

module.exports = {
  API_BASE,
  DEV_API_BASE,
  PROD_API_BASE,
  FORCE_PROD_API
};
