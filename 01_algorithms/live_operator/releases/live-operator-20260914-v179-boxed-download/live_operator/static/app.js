"use strict";

const MAX_OVERLAY_AGE_SECONDS = 0.75;
const STATUS_POLL_INTERVAL_MS = 500;
const LIVE_EVENT_POLL_INTERVAL_MS = 500;
const HISTORY_EVENT_POLL_INTERVAL_MS = 15000;
const LOW_FPS_RATIO = 0.75;
const LOW_FPS_CONSECUTIVE_SAMPLES = 3;
const REVIEW_RESULTS = new Set(["pending", "confirmed", "false_positive"]);
const CONFIRMED_REVIEW_CATEGORIES = new Set(["phone_use", "screen_capture"]);
const VLM_FILTER_RESULTS = new Set(["pass", "filter", "pending", "uncertain", "error"]);
const FALSE_POSITIVE_REASONS = new Set([
  "fixed_phone",
  "model_misdetect",
  "non_screen_use",
  "phone_call",
  "other",
]);
const RISK_BANDS = ["high", "medium", "low"];
const REVIEW_LABELS = {
  pending: "待复核",
  confirmed: "确认报警",
  false_positive: "已标误报",
};
const CONFIRMED_REVIEW_CATEGORY_LABELS = {
  phone_use: "使用手机",
  screen_capture: "拍摄屏幕",
};
const BEIJING_TIME_FORMATTER = new Intl.DateTimeFormat("en-US", {
  timeZone: "Asia/Shanghai",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  hourCycle: "h23",
});

const EVENT_TIME_FORMATTER = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai",
  month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
  hour12: false,
});

function reconcileRenderedItems(previous, items, keyOf, signatureOf, create) {
  const next = new Map();
  items.forEach(item => {
    const key = keyOf(item);
    const signature = signatureOf(item);
    const old = previous.get(key);
    next.set(key, old && old.signature === signature ? old : {signature, node: create(item)});
  });
  return next;
}

function formatTimestamp(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return String(value);
  return EVENT_TIME_FORMATTER.format(date);
}

function createGenerationGate() {
  let current = 0;
  return {
    begin() {
      current += 1;
      return current;
    },
    isCurrent(generation) {
      return generation === current;
    },
  };
}

function eventMediaKey(event) {
  if (!event) return "";
  return `${event.event_id || ""}|${event.clip_url || ""}|${event.overlay_url || ""}`;
}

async function prepareBoxedDownload(url, {request, onState, isCurrent, wait}) {
  for (let attempt = 0; attempt <= 120; attempt += 1) {
    if (!isCurrent()) return null;
    const payload = await request(url, attempt === 0 ? "POST" : "GET");
    if (!isCurrent()) return null;
    onState(payload.state);
    if (payload.state === "ready") {
      if (payload.download_url !== `${url}/file`) throw new Error("下载地址无效");
      return payload.download_url;
    }
    if (!["queued", "running"].includes(payload.state)) {
      throw new Error(payload.error || "视频生成失败，请重试");
    }
    if (attempt < 120) await wait(1000);
  }
  throw new Error("视频仍在生成，请稍后重试");
}

function selectOverlaySamples(timeline, currentTime, maxAge = MAX_OVERLAY_AGE_SECONDS) {
  if (!Array.isArray(timeline)) return [];
  const nearestTime = timeline.reduce((nearest, item) => {
    const sampleTime = Number(item && item.time_sec);
    if (!Number.isFinite(sampleTime) || sampleTime > currentTime) return nearest;
    return nearest === null || sampleTime > nearest ? sampleTime : nearest;
  }, null);
  if (nearestTime === null || currentTime - nearestTime > maxAge) return [];
  return timeline.filter((item) => Number(item && item.time_sec) === nearestTime);
}

function selectOverlaySample(timeline, currentTime, maxAge = MAX_OVERLAY_AGE_SECONDS) {
  return selectOverlaySamples(timeline, currentTime, maxAge)[0] || null;
}

function overlayBoxesForSample(sample) {
  if (!sample || typeof sample !== "object") return [];
  const inheritedFrameSize = {};
  if (Number.isFinite(Number(sample.frame_width))) inheritedFrameSize.frame_width = Number(sample.frame_width);
  if (Number.isFinite(Number(sample.frame_height))) inheritedFrameSize.frame_height = Number(sample.frame_height);
  const contextBoxes = Array.isArray(sample.boxes)
    ? sample.boxes
      .filter((item) => item && typeof item === "object")
      .map((item) => ({ ...inheritedFrameSize, ...item }))
    : Array.isArray(sample.bbox)
      ? [{
        ...inheritedFrameSize,
        label: sample.alarm === false ? "risk" : "person",
        bbox: sample.bbox,
        track_id: sample.track_id,
      }]
      : [];
  const phoneBoxes = Array.isArray(sample.phone_boxes)
    ? sample.phone_boxes
      .filter((item) => item && typeof item === "object" && Array.isArray(item.box || item.bbox))
      .map((item) => ({
        ...inheritedFrameSize,
        ...item,
        label: "phone",
        bbox: item.box || item.bbox,
      }))
    : [];
  return [...contextBoxes, ...phoneBoxes];
}

function eventLevel(event) {
  const level = String((event && (event.level || event.event_level)) || "risk").toLowerCase();
  return level === "alarm" || level === "review" ? level : "risk";
}

function eventIsAlarm(event) {
  return eventLevel(event) === "alarm";
}

function reviewResult(event) {
  const result = typeof event === "string" ? event : event && event.review_result;
  return REVIEW_RESULTS.has(result) ? result : "pending";
}

function reviewCategory(event) {
  const category = event && typeof event === "object" ? event.review_category : "";
  return CONFIRMED_REVIEW_CATEGORIES.has(category) ? category : "";
}

function reviewLabel(value) {
  const result = reviewResult(value);
  const category = result === "confirmed" ? reviewCategory(value) : "";
  return category ? CONFIRMED_REVIEW_CATEGORY_LABELS[category] : REVIEW_LABELS[result];
}

function vlmFilterResult(event) {
  const result = event && event.vlm_filter_result;
  return VLM_FILTER_RESULTS.has(result) ? result : "";
}

function vlmBadgePresentation(event) {
  const result = vlmFilterResult(event);
  if (result === "pass") return { className: "pass", label: "大模型通过" };
  if (result === "uncertain") return { className: "uncertain", label: "大模型不确定" };
  if (result === "pending") return { className: "pending", label: "大模型复核中" };
  if (result === "error") return { className: "error", label: "大模型复核异常" };
  if (result === "filter") return { className: "filter", label: "大模型判误报" };
  return { className: "missing", label: "大模型未复核" };
}

function vlmReviewSummaryParts(summary) {
  const source = summary && typeof summary === "object" ? summary : {};
  const count = (key) => {
    const value = Number(source[key]);
    return Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
  };
  return [
    { key: "reviewing", className: "", label: `大模型复核中 ${count("reviewing")} 条`, count: count("reviewing") },
    { key: "errors", className: "error", label: `复核异常 ${count("errors")} 条`, count: count("errors") },
  ].filter((item) => item.count > 0);
}

function updateEventReview(eventList, eventId, result, reason = null, category = null) {
  if (!REVIEW_RESULTS.has(result)) throw new Error("invalid review result");
  if (category !== null && (result !== "confirmed" || !CONFIRMED_REVIEW_CATEGORIES.has(category))) {
    throw new Error("invalid confirmed review category");
  }
  return (Array.isArray(eventList) ? eventList : []).map((event) => {
    if (!event || event.event_id !== eventId) return event;
    const updated = { ...event, review_result: result };
    if (result === "false_positive" && reason !== null) updated.review_reason = reason;
    else delete updated.review_reason;
    if (result === "confirmed" && category !== null) updated.review_category = category;
    else delete updated.review_category;
    return updated;
  });
}

function applyReviewOverrides(eventList, overrides) {
  return (Array.isArray(eventList) ? eventList : []).map((event) => {
    if (!event || !(overrides instanceof Map)) return event;
    const override = overrides.get(event.event_id);
    const result = typeof override === "string" ? override : override && override.result;
    if (!REVIEW_RESULTS.has(result)) return event;
    const updated = { ...event, review_result: result };
    const category = typeof override === "object" && override ? override.category : null;
    if (typeof override === "object" && override) {
      if (result === "confirmed" && CONFIRMED_REVIEW_CATEGORIES.has(category)) {
        updated.review_category = category;
      } else {
        delete updated.review_category;
      }
    }
    return updated;
  });
}

// Apply only the delta from this response's original rows. Server totals cover
// all pages; recomputing totals from the visible page would undercount them.
function reviewAdjustedSummary(summary, originalEvents, overrides, state) {
  const adjusted = { ...summary };
  const contribution = (event) => {
    const result = reviewResult(event);
    const category = reviewCategory(event);
    const visible = state.screenCaptureOnly
      ? result === "confirmed" && category === "screen_capture"
      : (state.showFalsePositives || result !== "false_positive")
        && (!state.excludeVlmFiltered || result === "confirmed" || vlmFilterResult(event) === "pass");
    return {
      events: visible ? 1 : 0,
      pending_review: visible && result === "pending" ? 1 : 0,
      confirmed_phone_use: visible && result === "confirmed" && category === "phone_use" ? 1 : 0,
      confirmed_screen_capture: visible && result === "confirmed" && category === "screen_capture" ? 1 : 0,
    };
  };
  originalEvents.forEach(event => {
    if (!overrides.has(event.event_id)) return;
    const before = contribution(event);
    const after = contribution(applyReviewOverrides([event], overrides)[0]);
    Object.keys(before).forEach(key => {
      if (Number.isFinite(adjusted[key])) adjusted[key] += after[key] - before[key];
    });
  });
  return adjusted;
}

function reconcileReviewOverrides(eventList, overrides) {
  const remaining = new Map(overrides instanceof Map ? overrides : []);
  (Array.isArray(eventList) ? eventList : []).forEach((event) => {
    if (!event || !remaining.has(event.event_id)) return;
    const override = remaining.get(event.event_id);
    const result = typeof override === "string" ? override : override && override.result;
    const category = typeof override === "object" && override ? override.category || "" : "";
    if (
      reviewResult(event) === result
      && (typeof override === "string" || reviewCategory(event) === category)
    ) {
      remaining.delete(event.event_id);
    }
  });
  return remaining;
}

function reviewSavePending(pendingReviews, eventOrId) {
  const eventId = typeof eventOrId === "object" && eventOrId
    ? eventOrId.event_id
    : eventOrId;
  return pendingReviews instanceof Map
    && typeof eventId === "string"
    && pendingReviews.has(eventId);
}

function createReviewRequest(eventOrId, result, reason = null, category = null) {
  if (!REVIEW_RESULTS.has(result)) throw new Error("invalid review result");
  if (
    result === "false_positive"
    && reason !== null
    && !FALSE_POSITIVE_REASONS.has(reason)
  ) {
    throw new Error("invalid false-positive reason");
  }
  if (category !== null && (result !== "confirmed" || !CONFIRMED_REVIEW_CATEGORIES.has(category))) {
    throw new Error("invalid confirmed review category");
  }
  const event = typeof eventOrId === "object" && eventOrId ? eventOrId : null;
  const eventId = event ? event.event_id : eventOrId;
  if (typeof eventId !== "string") throw new Error("invalid event id");
  const reviewUrl = event && typeof event.review_url === "string" && event.review_url
    ? event.review_url
    : `/api/events/${encodeURIComponent(eventId)}/review`;
  return {
    url: reviewUrl,
    options: {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        result === "false_positive" && reason !== null
          ? { result, reason }
          : result === "confirmed" && category !== null
            ? { result, category }
            : { result }
      ),
    },
  };
}

async function reviewRequestError(response) {
  let message = `HTTP ${response.status}`;
  try {
    const payload = await response.json();
    if (payload && typeof payload.error === "string" && payload.error) message = payload.error;
  } catch (_error) {
    // A proxy may send a non-JSON error page; keep the HTTP status in that case.
  }
  return new Error(message);
}

function reviewSaveErrorMessage(error) {
  const reason = error instanceof Error ? error.message : "";
  if (reason === "false-positive archive failed") {
    return "误报视频已归档，但预标注图片生成失败，请重试";
  }
  if (reason === "review save failed") return "复核结果写入失败，请重试";
  if (reason === "event is not reviewable") return "该事件的视频尚未准备好，请稍后重试";
  return "复核保存失败，请重试";
}

function isReadyEvent(event) {
  return Boolean(
    event
    && event.status === "ready"
    && event.clip_url
    && event.overlay_url
  );
}

function beijingTimeParts(value) {
  if (value === null || value === undefined || value === "") return null;
  const date = value instanceof Date ? value : new Date(value);
  if (!Number.isFinite(date.getTime())) return null;
  return Object.fromEntries(
    BEIJING_TIME_FORMATTER.formatToParts(date)
      .filter((part) => part.type !== "literal")
      .map((part) => [part.type, part.value]),
  );
}

function beijingDateKey(value) {
  const parts = beijingTimeParts(value);
  return parts ? `${parts.year}-${parts.month}-${parts.day}` : "";
}

function beijingHourKey(value) {
  const parts = beijingTimeParts(value);
  return parts ? `${parts.year}-${parts.month}-${parts.day}T${parts.hour}` : "";
}

function selectVisibleEvent(visibleEvents, selectedEventId) {
  if (!Array.isArray(visibleEvents) || !visibleEvents.length) return null;
  return visibleEvents.find((event) => event.event_id === selectedEventId) || visibleEvents[0];
}

function createDashboardState(now = new Date()) {
  return {
    selectedViews: [],
    selectedDate: "",
    selectedHour: "",
    calendarMonth: beijingDateKey(now).slice(0, 7),
    page: 1,
    pageSize: 20,
    followLatest: true,
    riskBands: ["high", "medium"],
    showFalsePositives: false,
    excludeVlmFiltered: true,
    screenCaptureOnly: false,
    screenCaptureRange: "all",
    selectedId: null,
  };
}

function buildEventsQuery(state) {
  const query = new URLSearchParams();
  query.set("page", String(state.page));
  query.set("page_size", String(state.pageSize));
  if (state.screenCaptureOnly) {
    query.set("screen_capture_only", "1");
    query.set("month", state.calendarMonth);
    query.set("review_range", state.screenCaptureRange || "all");
    if (state.selectedDate) query.set("date", state.selectedDate);
    return `/api/events?${query.toString()}`;
  }
  query.set("month", state.calendarMonth);
  selectedViewsForState(state).forEach((camera) => query.append("camera", camera));
  const selectedRiskBands = RISK_BANDS.filter((band) => Array.isArray(state.riskBands) && state.riskBands.includes(band));
  query.set("risk", selectedRiskBands.join(","));
  query.set("include_false_positives", state.showFalsePositives ? "1" : "0");
  query.set("exclude_vlm_filtered", state.excludeVlmFiltered ? "1" : "0");
  if (!state.followLatest) {
    if (state.selectedDate) query.set("date", state.selectedDate);
    if (state.selectedHour) query.set("hour", state.selectedHour);
  }
  return `/api/events?${query.toString()}`;
}

function toggleRiskBand(state, band) {
  if (!RISK_BANDS.includes(band)) throw new Error("invalid risk band");
  const selected = new Set(Array.isArray(state.riskBands) ? state.riskBands : []);
  if (selected.has(band)) selected.delete(band);
  else selected.add(band);
  return {
    ...state,
    riskBands: RISK_BANDS.filter((candidate) => selected.has(candidate)),
    page: 1,
    selectedId: null,
  };
}

function toggleFalsePositiveVisibility(state, showFalsePositives) {
  return {
    ...state,
    showFalsePositives: Boolean(showFalsePositives),
    page: 1,
    selectedId: null,
  };
}

function toggleVlmFilteredExclusion(state, excludeVlmFiltered) {
  return {
    ...state,
    excludeVlmFiltered: Boolean(excludeVlmFiltered),
    page: 1,
    selectedId: null,
  };
}

function toggleScreenCaptureOnly(state, screenCaptureOnly) {
  const enabled = Boolean(screenCaptureOnly);
  return {
    ...state,
    selectedViews: enabled ? [] : state.selectedViews,
    selectedDate: "",
    selectedHour: "",
    screenCaptureOnly: enabled,
    screenCaptureRange: "all",
    page: 1,
    followLatest: !enabled,
    selectedId: null,
  };
}

function selectedViewsForState(state) {
  const values = Array.isArray(state && state.selectedViews)
    ? state.selectedViews
    : (state && state.selectedView ? [state.selectedView] : []);
  return [...new Set(values.filter((value) => typeof value === "string" && value))];
}

function selectView(state, selectedView) {
  const selected = new Set(selectedViewsForState(state));
  if (!selectedView) selected.clear();
  else if (selected.has(selectedView)) selected.delete(selectedView);
  else selected.add(selectedView);
  return {
    ...state,
    selectedViews: [...selected],
    selectedDate: "",
    selectedHour: "",
    page: 1,
    followLatest: true,
    selectedId: null,
  };
}

function selectDate(state, selectedDate) {
  return {
    ...state,
    selectedDate,
    selectedHour: "",
    screenCaptureRange: state.screenCaptureOnly ? "date" : state.screenCaptureRange,
    page: 1,
    followLatest: false,
    selectedId: null,
  };
}

function selectScreenCaptureRange(state, screenCaptureRange) {
  if (!["all", "today", "7d"].includes(screenCaptureRange)) {
    throw new Error("invalid screen capture range");
  }
  return {
    ...state,
    selectedDate: "",
    selectedHour: "",
    screenCaptureRange,
    page: 1,
    followLatest: false,
    selectedId: null,
  };
}

function selectHour(state, selectedHour) {
  return {
    ...state,
    selectedHour,
    page: 1,
    followLatest: false,
    selectedId: null,
  };
}

function selectPage(state, page, selection) {
  return {
    ...state,
    selectedDate: selection.date || "",
    selectedHour: selection.hour || "",
    screenCaptureRange: selection.review_range || state.screenCaptureRange,
    page,
    followLatest: false,
    selectedId: null,
  };
}

function selectCalendarMonth(state, calendarMonth) {
  return { ...state, calendarMonth };
}

function applyEventsResponse(state, payload, preserveSelected) {
  const events = Array.isArray(payload && payload.events) ? payload.events : [];
  const pagination = payload && payload.pagination ? payload.pagination : {
    page: 1, page_size: state.pageSize, total: 0, total_pages: 0,
  };
  const selection = payload && payload.selection ? payload.selection : {};
  const responseSelectedViews = Array.isArray(selection.cameras)
    ? selection.cameras
    : selection.camera === undefined
      ? selectedViewsForState(state)
      : (selection.camera ? [selection.camera] : []);
  const selected = selectVisibleEvent(events, preserveSelected ? state.selectedId : null);
  return {
    state: {
      ...state,
      selectedViews: [...new Set(responseSelectedViews.filter((value) => typeof value === "string" && value))],
      selectedDate: selection.date || "",
      selectedHour: selection.hour || "",
      screenCaptureRange: selection.review_range || state.screenCaptureRange,
      calendarMonth: selection.month || state.calendarMonth,
      page: pagination.page,
      pageSize: pagination.page_size,
      selectedId: selected ? selected.event_id : null,
    },
    events,
    pagination,
    calendar: payload && payload.calendar ? payload.calendar : {},
    hours: payload && payload.hours ? payload.hours : {},
    dailyTotal: Number.isInteger(Number(selection.daily_total))
      ? Number(selection.daily_total)
      : 0,
    summary: payload && payload.summary ? payload.summary : {
      events: 0, alarms: 0, views: 0, max_risk: 0,
    },
    vlmReview: payload && payload.vlm_review ? payload.vlm_review : {
      reviewing: 0, errors: 0, uncertain: 0,
    },
  };
}

function pollIntervalForState(state) {
  return state.followLatest
    ? LIVE_EVENT_POLL_INTERVAL_MS
    : HISTORY_EVENT_POLL_INTERVAL_MS;
}

function createPollingLoop(run, interval, scheduler = {}) {
  const setTimer = scheduler.setTimer || ((callback, delay) => setTimeout(callback, delay));
  const clearTimer = scheduler.clearTimer || ((timer) => clearTimeout(timer));
  let timer = null;
  let inFlight = false;
  let rerunRequested = false;
  let stopped = false;

  function schedule(delay) {
    timer = setTimer(() => {
      timer = null;
      void tick();
    }, delay);
  }

  async function tick() {
    if (stopped || inFlight) return;
    inFlight = true;
    try {
      await run();
    } finally {
      inFlight = false;
      if (stopped) return;
      const delay = rerunRequested ? 0 : interval();
      rerunRequested = false;
      schedule(delay);
    }
  }

  return {
    refresh() {
      if (stopped) return false;
      if (timer !== null) {
        clearTimer(timer);
        timer = null;
      }
      if (inFlight) {
        rerunRequested = true;
        return false;
      }
      void tick();
      return true;
    },
    stop() {
      stopped = true;
      rerunRequested = false;
      if (timer !== null) clearTimer(timer);
      timer = null;
    },
    isInFlight() {
      return inFlight;
    },
  };
}

function paginationLabel(pagination) {
  const totalPages = Number(pagination && pagination.total_pages) || 0;
  const page = totalPages ? Number(pagination.page) || 1 : 0;
  const total = Number(pagination && pagination.total) || 0;
  return `第 ${page}/${totalPages} 页，共 ${total} 条`;
}

function eventCameraKey(event) {
  return (event && (event.camera || event.relay || event.view)) || "";
}

function buildDateCounts(events) {
  const counts = new Map();
  (Array.isArray(events) ? events : []).forEach((event) => {
    if (!isReadyEvent(event)) return;
    const dateKey = beijingDateKey(event.occurred_at);
    if (dateKey) counts.set(dateKey, (counts.get(dateKey) || 0) + 1);
  });
  return Object.fromEntries([...counts.entries()].sort(([left], [right]) => left.localeCompare(right)));
}

function availableHoursForDate(events, selectedDate) {
  const hours = new Set();
  (Array.isArray(events) ? events : []).forEach((event) => {
    if (!isReadyEvent(event) || beijingDateKey(event.occurred_at) !== selectedDate) return;
    const hourKey = beijingHourKey(event.occurred_at);
    if (hourKey) hours.add(hourKey);
  });
  return [...hours].sort();
}

function resolveTimeSelection(events, selectedDate, selectedHour) {
  const dates = Object.keys(buildDateCounts(events));
  const nextDate = dates.includes(selectedDate) ? selectedDate : (dates[dates.length - 1] || "");
  const hours = availableHoursForDate(events, nextDate);
  const nextHour = hours.includes(selectedHour) ? selectedHour : (hours[hours.length - 1] || "");
  return { selectedDate: nextDate, selectedHour: nextHour };
}

function filterEventsBySelection(events, selectedViews, selectedDate, selectedHour) {
  const selected = new Set(Array.isArray(selectedViews) ? selectedViews : (selectedViews ? [selectedViews] : []));
  return (Array.isArray(events) ? events : []).filter((event) => {
    if (!isReadyEvent(event)) return false;
    const dateKey = beijingDateKey(event.occurred_at);
    const hourKey = beijingHourKey(event.occurred_at);
    if (!dateKey || !hourKey) return false;
    return (!selected.size || selected.has(eventCameraKey(event)))
      && (!selectedDate || dateKey === selectedDate)
      && (!selectedHour || hourKey === selectedHour);
  });
}

function eventDisplayId(event) {
  const value = String((event && event.display_id) || "");
  return /^\d{12}$/.test(value) ? value : "编号待生成";
}

function dailyTotalLabel(total, selectedDate) {
  const value = Number(total);
  return selectedDate && Number.isInteger(value) && value > 0
    ? `当日共 ${value} 个事件`
    : "当日暂无事件";
}

function createHealthState() {
  return {
    cameras: [],
    backendAlerts: [],
    lowFpsCounts: {},
    dashboardReachable: true,
    statusFailures: 0,
    targetFpsPerStream: 0,
  };
}

function healthCameraKey(camera) {
  if (!camera || typeof camera !== "object") return "";
  return String(camera.relay || camera.view || "");
}

function validPositiveNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : 0;
}

function applyHealthSample(state, payload) {
  const current = state && typeof state === "object" ? state : createHealthState();
  const sample = payload && typeof payload === "object" ? payload : {};
  const cameras = Array.isArray(sample.cameras)
    ? sample.cameras.filter((camera) => camera && typeof camera === "object").map((camera) => ({ ...camera }))
    : [];
  const targetFpsPerStream = validPositiveNumber(sample.target_fps_per_stream);
  const previousCounts = current.lowFpsCounts && typeof current.lowFpsCounts === "object"
    ? current.lowFpsCounts
    : {};
  const lowFpsCounts = {};
  cameras.forEach((camera) => {
    const key = healthCameraKey(camera);
    const fps = Number(camera.fps);
    const isSlow = key
      && camera.status === "online"
      && targetFpsPerStream > 0
      && Number.isFinite(fps)
      && fps < targetFpsPerStream * LOW_FPS_RATIO;
    if (isSlow) lowFpsCounts[key] = (Number(previousCounts[key]) || 0) + 1;
  });
  const backendAlerts = Array.isArray(sample.alerts)
    ? sample.alerts
      .filter((alert) => alert && typeof alert === "object")
      .map((alert) => ({ ...alert }))
    : [];
  return {
    cameras,
    backendAlerts,
    lowFpsCounts,
    dashboardReachable: true,
    statusFailures: 0,
    targetFpsPerStream,
  };
}

function applyStatusFailure(state) {
  const current = state && typeof state === "object" ? state : createHealthState();
  return {
    cameras: Array.isArray(current.cameras)
      ? current.cameras.map((camera) => ({ ...camera }))
      : [],
    backendAlerts: Array.isArray(current.backendAlerts)
      ? current.backendAlerts.map((alert) => ({ ...alert }))
      : [],
    lowFpsCounts: { ...(current.lowFpsCounts || {}) },
    dashboardReachable: false,
    statusFailures: Math.min(3, (Number(current.statusFailures) || 0) + 1),
    targetFpsPerStream: validPositiveNumber(current.targetFpsPerStream),
  };
}

function statusFailurePresentation(state) {
  return Number(state && state.statusFailures) >= 3
    ? { text: "离线", className: "run-state is-offline" }
    : { text: "连接波动", className: "run-state is-connecting" };
}

function cameraHealthText(camera, isLowFps) {
  if (Number(camera && camera.source_errors) > 0) return "拉流错误";
  if (!camera || camera.status !== "online") return "离线";
  const fps = Number(camera.fps);
  if (!Number.isFinite(fps)) return "在线";
  return `${isLowFps ? "低速" : "在线"} · ${Math.round(fps)} FPS`;
}

function healthPresentation(state) {
  const current = state && typeof state === "object" ? state : createHealthState();
  const cameras = Array.isArray(current.cameras) ? current.cameras : [];
  const lowFpsCounts = current.lowFpsCounts && typeof current.lowFpsCounts === "object"
    ? current.lowFpsCounts
    : {};
  const alerts = (Array.isArray(current.backendAlerts) ? current.backendAlerts : [])
    .filter((alert) => (
      alert.severity === "critical" || alert.severity === "warning" || alert.severity === "info"
    ))
    .map((alert) => ({
      ...alert,
      code: String(alert.code || "health_warning"),
      message: String(alert.message || "系统状态异常"),
    }));
  const cameraLabels = {};
  cameras.forEach((camera) => {
    const key = healthCameraKey(camera);
    if (!key) return;
    const isLowFps = Number(lowFpsCounts[key]) >= LOW_FPS_CONSECUTIVE_SAMPLES;
    cameraLabels[key] = cameraHealthText(camera, isLowFps);
    if (isLowFps) {
      alerts.push({
        code: "camera_low_fps",
        severity: "warning",
        message: `${key} 持续低速`,
        camera: key,
      });
    }
  });
  if (current.dashboardReachable === false) {
    const persistent = Number(current.statusFailures) >= 3;
    alerts.push({
      code: persistent ? "dashboard_unreachable" : "dashboard_connection_fluctuation",
      severity: persistent ? "critical" : "warning",
      message: persistent ? "事件看板连续连接失败，请检查服务或网络" : "连接短暂波动，正在重试；当前为上次获取的状态",
    });
  }
  alerts.sort((left, right) => (
    (left.severity === "critical" ? 0 : left.severity === "warning" ? 1 : 2)
    - (right.severity === "critical" ? 0 : right.severity === "warning" ? 1 : 2)
  ));
  const severity = alerts.some((alert) => alert.severity === "critical")
    ? "critical"
    : alerts.some((alert) => alert.severity === "warning")
      ? "warning"
      : alerts.length
        ? "info"
      : "none";
  return {
    alerts,
    key: alerts.map((alert) => `${alert.severity}:${alert.code}:${alert.camera || ""}:${alert.message}`).join("|"),
    cameraLabels,
    hidden: alerts.length === 0,
    severity,
    summary: severity === "critical" ? "系统存在严重异常" : severity === "warning" ? "系统需要关注" : severity === "info" ? "系统已恢复" : "",
    details: alerts.map((alert) => alert.message).join("；"),
  };
}

function renderHealthAlerts(elements, presentation, dismissedKey = "") {
  const view = presentation || healthPresentation(createHealthState());
  const hidden = view.hidden || (view.key && view.key === dismissedKey);
  elements.bar.hidden = hidden;
  elements.bar.className = `system-alerts${hidden ? "" : ` is-${view.severity}`}`;
  elements.summary.textContent = view.summary;
  elements.details.textContent = view.details;
}

function nvrChannelFromPath(path) {
  const match = String(path || "").trim().match(/^\/Streaming\/Channels\/([1-9]\d*)01$/i);
  return match ? match[1] : "";
}

function nvrRtspPathFromChannel(channel) {
  const normalized = String(channel || "")
    .trim()
    .replace(/^(?:d|ch(?:annel)?|通道)\s*/i, "");
  if (!/^[1-9]\d{0,2}$/.test(normalized)) return "";
  return `/Streaming/Channels/${normalized}01`;
}

function normalizedCalibrationName(relay) {
  const safe = String(relay || "new_camera")
    .toLowerCase()
    .replace(/[^a-z0-9_.-]+/g, "_")
    .replace(/^[_-]+|[_-]+$/g, "") || "new_camera";
  return `${safe}_screen_calibration.json`;
}

function evidenceNumber(samples, field) {
  return Math.max(
    ...samples.map((sample) => Number(sample && sample[field])).filter(Number.isFinite),
    0,
  );
}

function evidenceStateRank(value) {
  const match = /^S(\d+)_/.exec(String(value || ""));
  return match ? Number(match[1]) : 0;
}

function medianNumber(values) {
  const sorted = values.filter(Number.isFinite).sort((left, right) => left - right);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function decisionEvidence(event, timeline) {
  const samples = Array.isArray(timeline)
    ? timeline.filter((sample) => sample && typeof sample === "object")
    : [];
  const phoneBoxes = samples.flatMap((sample) => Array.isArray(sample.phone_boxes) ? sample.phone_boxes : []);
  // `phone_score` is an internal reliability feature that intentionally
  // saturates at 1.0 for normal-sized high-confidence boxes.  It is not the
  // PhoneDet output and must never be presented as the detector confidence.
  const phoneConfidences = phoneBoxes
    .map((box) => Number(box && box.confidence))
    .filter(Number.isFinite);
  const medianPhoneConfidence = medianNumber(phoneConfidences);
  const maxPhoneConfidence = phoneConfidences.length ? Math.max(...phoneConfidences) : null;
  const states = [...new Set(samples.map((sample) => String(sample.state || "")).filter(Boolean))];
  const strongestState = states.sort((left, right) => evidenceStateRank(right) - evidenceStateRank(left))[0]
    || String((event && event.detector_state) || "");
  const stateRank = evidenceStateRank(strongestState);
  const screenIds = [...new Set(samples.map((sample) => String(sample.screen_id || "")).filter(Boolean))];
  const stableHandheld = evidenceNumber(samples, "handheld_phone_stable_count");
  const windowHits = evidenceNumber(samples, "window_hits");
  const rejectReasons = [...new Set(samples.map((sample) => String(sample.gated_reject_reason || "")).filter(Boolean))];
  const staticSuppressed = samples.some((sample) => sample.static_suppressed === true);
  const risk = Number(event && event.risk_peak);
  const hasDetailedEvidence = samples.some((sample) => (
    Object.prototype.hasOwnProperty.call(sample, "window_hits")
    || Object.prototype.hasOwnProperty.call(sample, "screen_id")
    || Object.prototype.hasOwnProperty.call(sample, "handheld_phone_stable_count")
  ));
  const isAlarm = eventIsAlarm(event) || stateRank >= 5;
  return {
    hasDetailedEvidence,
    isAlarm,
    steps: [
      {
        label: "手机检测",
        value: medianPhoneConfidence === null
          ? "未记录原始分数"
          : `原始中位 ${(medianPhoneConfidence * 100).toFixed(0)}%`,
        detail: maxPhoneConfidence === null
          ? "事件片段未带手机框"
          : `原始峰值 ${(maxPhoneConfidence * 100).toFixed(0)}%，${phoneBoxes.length} 个关联候选框`,
        state: medianPhoneConfidence === null ? "unknown" : "confirmed",
      },
      {
        label: "人员关联",
        value: stateRank >= 2 ? "已通过关联" : strongestState || "未记录",
        detail: stableHandheld ? `稳定关联 ${stableHandheld} 帧` : `状态 ${strongestState || "--"}`,
        state: stateRank >= 2 ? "confirmed" : "unknown",
      },
      {
        label: "屏幕关联",
        value: screenIds.length ? screenIds.join("、") : stateRank >= 3 ? "已进入屏幕证据阶段" : "未确认",
        detail: screenIds.length ? "命中已标定屏幕" : "未记录具体屏幕编号",
        state: screenIds.length || stateRank >= 3 ? "confirmed" : "unknown",
      },
      {
        label: "时序累计",
        value: windowHits ? `累计 ${windowHits} 帧` : isAlarm ? "已达到报警时序" : "未记录",
        detail: staticSuppressed ? "固定手机抑制已生效" : "同一人员连续证据累计",
        state: windowHits || isAlarm ? "confirmed" : "unknown",
      },
      {
        label: "最终判定",
        value: isAlarm ? "稳定报警" : strongestState || "待判断",
        detail: rejectReasons.length
          ? `拦截原因：${rejectReasons.join("、")}`
          : `峰值风险 ${Number.isFinite(risk) ? risk.toFixed(2) : "--"}`,
        state: isAlarm ? "alarm" : "unknown",
      },
    ],
  };
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    STATUS_POLL_INTERVAL_MS,
    LIVE_EVENT_POLL_INTERVAL_MS,
    HISTORY_EVENT_POLL_INTERVAL_MS,
    LOW_FPS_RATIO,
    LOW_FPS_CONSECUTIVE_SAMPLES,
    createGenerationGate,
    eventMediaKey,
    prepareBoxedDownload,
    selectOverlaySample,
    selectOverlaySamples,
    overlayBoxesForSample,
    eventLevel,
    eventIsAlarm,
    reviewResult,
    reviewCategory,
    reviewLabel,
    vlmFilterResult,
    vlmBadgePresentation,
    vlmReviewSummaryParts,
    updateEventReview,
    applyReviewOverrides,
    reviewAdjustedSummary,
    reconcileReviewOverrides,
    reviewSavePending,
    createReviewRequest,
    reviewSaveErrorMessage,
    isReadyEvent,
    beijingDateKey,
    beijingHourKey,
    formatTimestamp,
    reconcileRenderedItems,
    selectVisibleEvent,
    createDashboardState,
    buildEventsQuery,
    selectedViewsForState,
    selectView,
    selectDate,
    selectScreenCaptureRange,
    selectHour,
    selectPage,
    selectCalendarMonth,
    toggleRiskBand,
    toggleFalsePositiveVisibility,
    toggleVlmFilteredExclusion,
    toggleScreenCaptureOnly,
    applyEventsResponse,
    eventDisplayId,
    dailyTotalLabel,
    pollIntervalForState,
    createPollingLoop,
    paginationLabel,
    buildDateCounts,
    availableHoursForDate,
    resolveTimeSelection,
    filterEventsBySelection,
    createHealthState,
    applyHealthSample,
    applyStatusFailure,
    statusFailurePresentation,
    healthPresentation,
    renderHealthAlerts,
    nvrChannelFromPath,
    nvrRtspPathFromChannel,
    normalizedCalibrationName,
    medianNumber,
    decisionEvidence,
  };
}

if (typeof document !== "undefined") {
  const cameraList = document.getElementById("camera-status");
  const systemAlerts = document.getElementById("system-alerts");
  const systemAlertSummary = document.getElementById("system-alert-summary");
  const systemAlertDetails = document.getElementById("system-alert-details");
  const systemAlertClose = document.getElementById("system-alert-close");
  const eventList = document.getElementById("latest-events");
  const eventFilterBand = document.querySelector(".event-filter-band");
  const eventsHeading = document.getElementById("events-heading");
  const viewFilters = document.getElementById("view-filters");
  const clearFilterButton = document.getElementById("clear-filter");
  const eventCalendar = document.getElementById("event-calendar");
  const calendarTrigger = document.getElementById("calendar-trigger");
  const calendarPopover = document.getElementById("calendar-popover");
  const calendarPrev = document.getElementById("calendar-prev");
  const calendarNext = document.getElementById("calendar-next");
  const calendarGrid = document.getElementById("calendar-grid");
  const calendarContextLabel = document.getElementById("calendar-context-label");
  const timeFilterLabel = document.getElementById("time-filter-label");
  const showFalsePositives = document.getElementById("show-false-positives");
  const excludeVlmFiltered = document.getElementById("exclude-vlm-filtered");
  const vlmFilterSummary = document.getElementById("vlm-filter-summary");
  const showScreenCaptures = document.getElementById("show-screen-captures");
  const hourFilters = document.getElementById("hour-filters");
  const eventPagePrev = document.getElementById("event-page-prev");
  const eventPageNext = document.getElementById("event-page-next");
  const videoShell = document.getElementById("video-shell");
  const videoStage = document.getElementById("video-stage");
  const video = document.getElementById("event-video");
  const canvas = document.getElementById("event-canvas");
  const fullscreenButton = document.getElementById("event-fullscreen");
  const downloadButton = document.getElementById("event-download");
  let downloadEvent = null;
  let downloadGeneration = 0;
  const fullscreenVideoControls = document.getElementById("fullscreen-video-controls");
  const fullscreenPlayToggle = document.getElementById("fullscreen-play-toggle");
  const fullscreenProgress = document.getElementById("fullscreen-progress");
  const fullscreenTime = document.getElementById("fullscreen-time");
  const emptyState = document.getElementById("empty-state");
  const showAlarmBoxes = document.getElementById("show-alarm-boxes");
  const showPhoneBoxes = document.getElementById("show-phone-boxes");
  const showContextBoxes = document.getElementById("show-context-boxes");
  const reviewActions = document.getElementById("review-actions");
  const reviewControls = [...reviewActions.querySelectorAll("[data-review-result]")];
  const reviewError = document.getElementById("review-error");
  const evidenceChainList = document.getElementById("evidence-chain-list");
  const evidenceChainStatus = document.getElementById("evidence-chain-status");
  const falsePositiveDialog = document.getElementById("false-positive-dialog");
  const falsePositiveReasonButtons = [
    ...falsePositiveDialog.querySelectorAll("[data-false-positive-reason]"),
  ];
  const fixedObjectManagerOpen = document.getElementById("fixed-object-manager-open");
  const fixedObjectDialog = document.getElementById("fixed-object-dialog");
  const fixedObjectList = document.getElementById("fixed-object-list");
  const fixedObjectEmpty = document.getElementById("fixed-object-empty");
  const fixedObjectError = document.getElementById("fixed-object-error");
  const cameraManagementOpen = document.getElementById("camera-management-open");
  const cameraManagementDialog = document.getElementById("camera-management-dialog");
  const cameraManagementList = document.getElementById("camera-management-list");
  const cameraManagementStatus = document.getElementById("camera-management-status");
  const cameraManagementSummary = document.getElementById("camera-management-summary");
  const cameraManagementFilters = document.getElementById("camera-management-filters");
  const cameraManagementAdd = document.getElementById("camera-management-add");
  const cameraManagementSave = document.getElementById("camera-management-save");
  const cameraManagementDiscard = document.getElementById("camera-management-discard");
  const cameraManagementPublish = document.getElementById("camera-management-publish");
  const screenCalibrationDialog = document.getElementById("screen-calibration-dialog");
  const screenCalibrationCamera = document.getElementById("screen-calibration-camera");
  const screenCalibrationStage = document.getElementById("screen-calibration-stage");
  const screenCalibrationImage = document.getElementById("screen-calibration-image");
  const screenCalibrationCanvas = document.getElementById("screen-calibration-canvas");
  const screenCalibrationLoading = document.getElementById("screen-calibration-loading");
  const screenCalibrationSelect = document.getElementById("screen-calibration-select");
  const screenCalibrationAdd = document.getElementById("screen-calibration-add");
  const screenCalibrationRemove = document.getElementById("screen-calibration-remove");
  const screenCalibrationReset = document.getElementById("screen-calibration-reset");
  const screenCalibrationSave = document.getElementById("screen-calibration-save");
  const screenCalibrationError = document.getElementById("screen-calibration-error");
  const reviewToast = document.getElementById("review-toast");
  const reviewToastMessage = document.getElementById("review-toast-message");
  const reviewToastUndo = document.getElementById("review-toast-undo");
  const refreshGate = createGenerationGate();
  const selectionGate = createGenerationGate();
  let fullscreenControlsHideTimer = null;
  let playbackProgressAnimationActive = false;
  let renderedEvidenceKey = "";
  let renderedReviewControlsKey = "";

  let events = [];
  let cameras = [];
  let healthState = createHealthState();
  let dismissedHealthAlertKey = "";
  let dashboardState = createDashboardState();
  let eventPagination = { page: 1, page_size: 20, total: 0, total_pages: 0 };
  let eventSelection = { camera: "", month: dashboardState.calendarMonth, date: "", hour: "" };
  let eventDailyTotal = 0;
  let calendarCounts = {};
  let hourCounts = {};
  let eventSummary = { events: 0, alarms: 0, views: 0, max_risk: 0 };
  let summaryOriginalEvents = [];
  let vlmReviewSummary = { reviewing: 0, errors: 0, uncertain: 0 };
  let selectedMediaKey = "";
  let overlayTimeline = [];
  let overlayController = null;
  let overlaySlots = [];
  let pendingRefreshPreservesSelection = true;
  const pendingReviews = new Map();
  let reviewOverrides = new Map();
  let eventPoller = null;
  let eventRequestController = null;
  let viewFilterOptionsKey = "";
  let renderedEventNodes = new Map();
  let renderedCameraNodes = new Map();
  let eventsLoaded = false;
  let userQueryPending = true;
  const eventColumn = document.querySelector(".event-column");
  const requestStatus = document.getElementById("event-request-status");
  let hourFilterOptionsKey = "";
  let reviewToastTimer = null;
  let undoReview = null;
  let cameraManagement = null;
  let cameraManagementBusy = false;
  let cameraManagementRestarting = false;
  let cameraManagementFilter = "all";
  let cameraManagementPreviewRequest = 0;
  const cameraManagementPreviewUrls = new Map();
  const screenCalibrationEdits = new Map();
  let screenCalibration = null;
  let draggedCalibrationVertex = null;
  let screenCalibrationPreviewRequest = 0;
  let screenCalibrationPreviewObjectUrl = "";

  const boxStyles = {
    person: { color: "#ef3f34", label: "疑似拍屏" },
    risk: { color: "#ff8a1f", label: "疑似拍屏" },
    phone: { color: "#ffb020", label: "手机" },
    screen: { color: "#388be0", label: "屏幕" },
  };

  async function fetchJson(url, signal, method = "GET") {
    const controller = new AbortController();
    let timedOut = false;
    const cancel = () => controller.abort();
    if (signal) {
      if (signal.aborted) cancel();
      else signal.addEventListener("abort", cancel, {once: true});
    }
    const timeout = setTimeout(() => { timedOut = true; cancel(); }, 12000);
    try {
    const response = await fetch(url, { method, cache: "no-store", signal: controller.signal });
    if (response.status === 401) {
      window.location.replace(`/login?next=${encodeURIComponent(window.location.pathname + window.location.search)}`);
      throw new Error("登录已过期");
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
    } catch (error) {
      if (timedOut) throw new Error("请求超时");
      throw error;
    } finally {
      clearTimeout(timeout);
      if (signal) signal.removeEventListener("abort", cancel);
    }
  }

  function setText(id, value) {
    const element = document.getElementById(id);
    if (element && element.textContent !== String(value)) element.textContent = String(value);
  }

  function cameraLabel(value) {
    const camera = cameras.find((item) => item.relay === value || item.view === value);
    return camera ? (camera.view || camera.relay || value) : (value || "未知视角");
  }

  function eventCamera(event) {
    return eventCameraKey(event);
  }

  function shiftMonth(monthKey, delta) {
    const match = /^(\d{4})-(\d{2})$/.exec(monthKey);
    if (!match) return monthKey;
    const shifted = new Date(Date.UTC(Number(match[1]), Number(match[2]) - 1 + delta, 1));
    return `${shifted.getUTCFullYear()}-${String(shifted.getUTCMonth() + 1).padStart(2, "0")}`;
  }

  function closeCalendar() {
    calendarPopover.hidden = true;
    calendarTrigger.setAttribute("aria-expanded", "false");
  }

  function renderCalendar() {
    const screenCaptureMode = dashboardState.screenCaptureOnly;
    setText(
      "selected-date-label",
      dashboardState.selectedDate
        ? `${dashboardState.selectedDate.slice(0, 4)}年${dashboardState.selectedDate.slice(5, 7)}月${dashboardState.selectedDate.slice(8, 10)}日`
        : screenCaptureMode ? "全部确认日期" : "暂无日期",
    );
    setText("daily-event-total", screenCaptureMode
      ? dashboardState.selectedDate
        ? `当日确认 ${eventDailyTotal} 条`
        : "点击日期查看当天记录"
      : dailyTotalLabel(eventDailyTotal, dashboardState.selectedDate));
    const monthKey = dashboardState.calendarMonth;
    const match = /^(\d{4})-(\d{2})$/.exec(monthKey);
    if (!match) {
      calendarGrid.replaceChildren();
      return;
    }
    const year = Number(match[1]);
    const month = Number(match[2]);
    setText("calendar-month-label", `${year}年${month}月`);
    const firstWeekday = (new Date(Date.UTC(year, month - 1, 1)).getUTCDay() + 6) % 7;
    const daysInMonth = new Date(Date.UTC(year, month, 0)).getUTCDate();
    const cells = [];
    for (let index = 0; index < 42; index += 1) {
      const day = index - firstWeekday + 1;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "calendar-day";
      if (day < 1 || day > daysInMonth) {
        button.classList.add("is-outside");
        button.disabled = true;
        button.tabIndex = -1;
        cells.push(button);
        continue;
      }
      const dateKey = `${monthKey}-${String(day).padStart(2, "0")}`;
      const count = calendarCounts[dateKey] || 0;
      const dayNumber = document.createElement("span");
      dayNumber.className = "calendar-day-number";
      dayNumber.textContent = String(day);
      button.append(dayNumber);
      button.disabled = count === 0;
      const itemName = screenCaptureMode ? "人工确认拍屏" : "报警";
      button.setAttribute("aria-label", count ? `${dateKey}，${count}条${itemName}` : `${dateKey}，无${itemName}`);
      if (count) {
        const countLabel = document.createElement("span");
        countLabel.className = "calendar-day-count";
        countLabel.textContent = `${count}条`;
        button.append(countLabel);
        button.addEventListener("click", () => {
          closeCalendar();
          changeDashboardState(selectDate(dashboardState, dateKey), false);
        });
      }
      if (dateKey === dashboardState.selectedDate) {
        button.classList.add("selected");
        button.setAttribute("aria-current", "date");
      }
      cells.push(button);
    }
    calendarGrid.replaceChildren(...cells);
  }

  function hourLabel(hourKey) {
    const hour = Number(hourKey.slice(-2));
    const nextHour = (hour + 1) % 24;
    return `${String(hour).padStart(2, "0")}:00-${String(nextHour).padStart(2, "0")}:00`;
  }

  function renderHourFilters() {
    const hours = Object.keys(hourCounts)
      .filter((hourKey) => hourKey.startsWith(`${dashboardState.selectedDate}T`))
      .sort();
    if (!hours.length) {
      if (hourFilterOptionsKey !== "empty") {
        const empty = document.createElement("span");
        empty.className = "hour-empty";
        empty.textContent = "暂无可用时段";
        hourFilters.replaceChildren(empty);
        hourFilterOptionsKey = "empty";
      }
      return;
    }
    const optionsKey = JSON.stringify(hours.map((hourKey) => [hourKey, hourCounts[hourKey]]));
    if (optionsKey !== hourFilterOptionsKey) {
      hourFilters.replaceChildren(...hours.map((hourKey) => {
        const button = document.createElement("button");
        const hour = hourKey.slice(-2);
        button.type = "button";
        button.dataset.hour = hour;
        button.className = "hour-button";
        button.textContent = `${hourLabel(hourKey)} · ${hourCounts[hourKey]}`;
        button.addEventListener("click", () => {
          changeDashboardState(selectHour(dashboardState, hour), false);
        });
        return button;
      }));
      hourFilterOptionsKey = optionsKey;
    }
    hourFilters.querySelectorAll("button[data-hour]").forEach((button) => {
      const isSelected = button.dataset.hour === dashboardState.selectedHour;
      button.classList.toggle("selected", isSelected);
      button.setAttribute("aria-pressed", String(isSelected));
    });
  }

  function renderScreenCaptureTimeFilters() {
    const options = [
      ["all", "全部"],
      ["today", "今天"],
      ["7d", "近7天"],
    ];
    hourFilterOptionsKey = "";
    hourFilters.replaceChildren(...options.map(([value, label]) => {
      const button = document.createElement("button");
      const isSelected = !dashboardState.selectedDate
        && dashboardState.screenCaptureRange === value;
      button.type = "button";
      button.className = `hour-button screen-capture-range-button ${isSelected ? "selected" : ""}`;
      button.setAttribute("aria-pressed", String(isSelected));
      button.textContent = label;
      button.addEventListener("click", () => {
        changeDashboardState(selectScreenCaptureRange(dashboardState, value), false);
      });
      return button;
    }));
  }

  function renderSystemAlerts() {
    const presentation = healthPresentation(healthState);
    if (presentation.hidden) dismissedHealthAlertKey = "";
    cameras = healthState.cameras;
    renderHealthAlerts(
      {
        bar: systemAlerts,
        summary: systemAlertSummary,
        details: systemAlertDetails,
      },
      presentation,
      dismissedHealthAlertKey,
    );
    return presentation;
  }

  systemAlertClose.addEventListener("click", () => {
    dismissedHealthAlertKey = healthPresentation(healthState).key;
    renderSystemAlerts();
  });

  function renderStatus(status) {
    healthState = applyHealthSample(healthState, status);
    const presentation = renderSystemAlerts();
    const state = String(status.state || "unknown");
    setText("run-id", status.run_id || "当前运行");
    setText("run-state", state === "running" ? "运行中" : state === "offline" ? "离线" : state);
    const updated = Number(status.updated_at);
    setText("run-updated", Number.isFinite(updated) ? new Date(updated * 1000).toLocaleString("zh-CN", { hour12: false }) : "实时状态");
    const online = cameras.filter((camera) => camera.status === "online").length;
    setText("camera-count", `${online}/${cameras.length} 路在线`);
    const stateElement = document.getElementById("run-state");
    stateElement.className = `run-state ${state === "running" ? "is-running" : state === "offline" ? "is-offline" : "is-waiting"}`;
    renderedCameraNodes = reconcileRenderedItems(renderedCameraNodes, cameras,
      camera => healthCameraKey(camera),
      camera => JSON.stringify([camera.view, camera.relay, camera.status, camera.source_errors,
        presentation.cameraLabels[healthCameraKey(camera)] || cameraHealthText(camera, false)]),
      camera => cameraRow(camera, presentation));
    syncRenderedChildren(cameraList, renderedCameraNodes);
    renderViewFilters();
  }

  function cameraRow(camera, presentation) {
    const row = document.createElement("div");
    row.className = "camera-row";
    const identity = document.createElement("div");
    const view = document.createElement("strong");
    view.textContent = camera.view || camera.relay || "摄像头";
    identity.append(view);
    const health = document.createElement("div");
    const cameraKey = healthCameraKey(camera);
    const label = presentation.cameraLabels[cameraKey] || cameraHealthText(camera, false);
    const stateClass = Number(camera.source_errors) > 0 || camera.status !== "online"
      ? "offline"
      : label.startsWith("低速")
        ? "warning"
        : "online";
    health.className = `camera-health ${stateClass}`;
    health.textContent = label;
    row.append(identity, health);
    return row;
  }

  function availableViews() {
    const labels = new Map();
    cameras.forEach((camera) => {
      if (camera.relay) labels.set(camera.relay, camera.view || camera.relay);
    });
    return [...labels.entries()];
  }

  function renderViewFilters() {
    const options = [["", "全部"], ...availableViews()];
    const selectedViews = new Set(selectedViewsForState(dashboardState));
    const optionsKey = JSON.stringify(options);
    if (optionsKey !== viewFilterOptionsKey) {
      let lastGroup = "";
      const nodes = [];
      options.forEach(([value, label]) => {
        const group = !value ? "" : label.startsWith("FFS") ? "FFS 区域" : label.startsWith("总工办") ? "总工办资料室" : "其他区域";
        if (group && group !== lastGroup) {
          const heading = document.createElement("div");
          heading.className = "camera-group-label";
          heading.textContent = group;
          nodes.push(heading);
        }
        lastGroup = group;
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.view = value;
        button.className = "filter-button";
        button.textContent = label;
        button.addEventListener("click", () => {
          changeDashboardState(selectView(dashboardState, value), false);
        });
        nodes.push(button);
      });
      viewFilters.replaceChildren(...nodes);
      viewFilterOptionsKey = optionsKey;
    }
    viewFilters.querySelectorAll("button[data-view]").forEach((button) => {
      const value = button.dataset.view || "";
      const isSelected = value ? selectedViews.has(value) : selectedViews.size === 0;
      button.classList.toggle("selected", isSelected);
      button.setAttribute("aria-pressed", String(isSelected));
      button.disabled = dashboardState.screenCaptureOnly;
    });
    clearFilterButton.hidden = dashboardState.screenCaptureOnly || selectedViews.size === 0;
    const selectionLabel = dashboardState.screenCaptureOnly
      ? "全部人工标注拍屏"
      : selectedViews.size === 0
      ? "全部视角"
      : selectedViews.size === 1
        ? cameraLabel([...selectedViews][0])
        : `已选 ${selectedViews.size} 个视角`;
    setText("event-filter-label", selectionLabel);
  }

  function renderVlmFilterSummary() {
    const parts = vlmReviewSummaryParts(vlmReviewSummary);
    const shouldShow = dashboardState.excludeVlmFiltered
      && !dashboardState.screenCaptureOnly
      && parts.length > 0;
    vlmFilterSummary.hidden = !shouldShow;
    vlmFilterSummary.replaceChildren(...parts.map((part) => {
      const badge = document.createElement("span");
      badge.className = part.className;
      badge.textContent = part.label;
      return badge;
    }));
  }

  function renderEvents() {
    events = applyReviewOverrides(events, reviewOverrides);
    const visibleSummary = reviewAdjustedSummary(eventSummary, summaryOriginalEvents, reviewOverrides, dashboardState);
    setText("event-count", visibleSummary.events || 0);
    setText("pending-review-count", visibleSummary.pending_review ?? "--");
    setText("phone-use-count", visibleSummary.confirmed_phone_use ?? "--");
    setText("screen-capture-count", visibleSummary.confirmed_screen_capture ?? "--");
    setText("view-count", eventSummary.views || 0);
    renderViewFilters();
    eventFilterBand.hidden = false;
    setText("events-heading", dashboardState.screenCaptureOnly ? "人工标注拍屏事件" : "检测时间段");
    calendarContextLabel.textContent = dashboardState.screenCaptureOnly ? "确认日期" : "报警日期";
    timeFilterLabel.textContent = dashboardState.screenCaptureOnly ? "确认时间范围" : "时间筛选";
    calendarPopover.setAttribute(
      "aria-label",
      dashboardState.screenCaptureOnly ? "选择人工确认日期" : "选择报警日期",
    );
    hourFilters.setAttribute(
      "aria-label",
      dashboardState.screenCaptureOnly ? "人工确认时间范围" : "报警时间筛选",
    );
    showFalsePositives.checked = dashboardState.showFalsePositives;
    showFalsePositives.disabled = dashboardState.screenCaptureOnly;
    excludeVlmFiltered.checked = dashboardState.excludeVlmFiltered;
    excludeVlmFiltered.disabled = dashboardState.screenCaptureOnly;
    renderVlmFilterSummary();
    showScreenCaptures.classList.toggle("is-active", dashboardState.screenCaptureOnly);
    showScreenCaptures.setAttribute("aria-pressed", String(dashboardState.screenCaptureOnly));
    showScreenCaptures.title = dashboardState.screenCaptureOnly
      ? "取消筛选并返回全部事件"
      : "查看全部由人工确认为拍摄屏幕的事件";
    renderCalendar();
    if (dashboardState.screenCaptureOnly) renderScreenCaptureTimeFilters();
    else renderHourFilters();
    renderPagination();

    const selected = selectVisibleEvent(events, dashboardState.selectedId);
    dashboardState = { ...dashboardState, selectedId: selected ? selected.event_id : null };
    renderedEventNodes = reconcileRenderedItems(renderedEventNodes, events,
      event => event.review_url || `${event.source_run_id || ""}/${event.event_id}`,
      event => JSON.stringify([event, dashboardState.selectedId === event.event_id,
        dashboardState.screenCaptureOnly, cameraLabel(eventCamera(event))]), eventButton);
    syncRenderedChildren(eventList, renderedEventNodes);
    const listEmpty = document.getElementById("event-empty");
    listEmpty.hidden = events.length > 0;
    listEmpty.textContent = dashboardState.excludeVlmFiltered
      && Number(vlmReviewSummary.reviewing || 0) > 0
      ? "当前时段事件正在大模型复核，通过后将自动显示"
      : "当前筛选页暂无符合条件的报警事件";
    if (selected) {
      updateEventSummary(selected);
      if (eventMediaKey(selected) !== selectedMediaKey) loadSelectedMedia(selected);
    } else {
      selectedMediaKey = "";
      resetReview();
    }
  }

  function renderPagination() {
    const page = Number(eventPagination.page) || 1;
    const totalPages = Number(eventPagination.total_pages) || 0;
    eventPagePrev.disabled = totalPages === 0 || page <= 1;
    eventPageNext.disabled = totalPages === 0 || page >= totalPages;
    setText("event-page-label", paginationLabel(eventPagination));
  }

  function syncRenderedChildren(container, entries) {
    const nodes = [...entries.values()].map(item => item.node);
    if (nodes.length !== container.children.length || nodes.some((node, i) => container.children[i] !== node)) {
      container.replaceChildren(...nodes);
    }
  }

  function showRequestState(message, busy, error = false) {
    if (requestStatus) {
      setText("event-request-message", message);
      requestStatus.classList.toggle("is-error", error);
      document.getElementById("event-retry").hidden = !error;
    }
    eventColumn.setAttribute("aria-busy", String(busy));
    eventColumn.classList.toggle("is-updating", busy && eventsLoaded);
  }

  function eventButton(event) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `event-row ${event.event_id === dashboardState.selectedId ? "selected" : ""}`;

    const top = document.createElement("div");
    top.className = "event-topline";
    const name = document.createElement("strong");
    name.className = "event-name";
    name.textContent = `${cameraLabel(eventCamera(event))} · ${formatTimestamp(event.occurred_at)}`;
    const badge = document.createElement("span");
    const level = eventLevel(event);
    const initialPass = level === "alarm" || level === "review";
    badge.className = `event-state ${initialPass ? "alarm" : level}`;
    badge.textContent = initialPass ? "初检通过" : "RISK";
    const badges = document.createElement("div");
    badges.className = "event-badges";
    badges.append(badge);
    const vlmPresentation = vlmBadgePresentation(event);
    if (vlmPresentation) {
      const vlmBadge = document.createElement("span");
      vlmBadge.className = `vlm-badge ${vlmPresentation.className}`;
      vlmBadge.textContent = vlmPresentation.label;
      badges.append(vlmBadge);
    }
    const result = reviewResult(event);
    if (result !== "pending") {
      const reviewBadge = document.createElement("span");
      reviewBadge.className = `review-badge ${result === "false_positive" ? "false-positive" : "confirmed"}`;
      reviewBadge.textContent = reviewLabel(event);
      badges.append(reviewBadge);
    }
    top.append(name, badges);

    let confirmation = null;
    if (dashboardState.screenCaptureOnly) {
      confirmation = document.createElement("div");
      const hasReviewTime = Boolean(event.reviewed_at);
      confirmation.className = `screen-capture-review-time ${hasReviewTime ? "" : "history"}`.trim();
      confirmation.textContent = hasReviewTime
        ? `人工确认 ${formatTimestamp(event.reviewed_at)} · 事件发生 ${formatTimestamp(event.occurred_at)}`
        : `历史记录 · 未保存确认时间 · 事件发生 ${formatTimestamp(event.occurred_at)}`;
    }

    const stats = document.createElement("div");
    stats.className = "event-stats";
    const values = [
      ["风险", Number(event.risk_peak || 0).toFixed(2)],
      ["状态", reviewLabel(event)],
      ["事件编号", eventDisplayId(event)],
    ];
    values.forEach(([label, value]) => {
      const cell = document.createElement("span");
      const strong = document.createElement("strong");
      cell.append(document.createTextNode(label), strong);
      strong.textContent = value;
      stats.append(cell);
    });

    const track = document.createElement("div");
    track.className = "risk-track";
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(0, Math.min(100, Number(event.risk_peak || 0) * 100))}%`;
    track.append(fill);
    button.append(top);
    if (confirmation) button.append(confirmation);
    button.append(stats, track);
    button.addEventListener("click", () => {
      selectEvent(event);
      // The event-row click is an explicit user gesture, so browsers allow
      // playback here without enabling unsolicited autoplay on page load or
      // during the 0.5-second event refresh.
      video.play().catch(() => {});
    });
    return button;
  }

  function selectEvent(event, rerender = true) {
    const changed = dashboardState.selectedId !== event.event_id;
    dashboardState = { ...dashboardState, selectedId: event.event_id };
    if (changed) {
      selectedMediaKey = "";
      setReviewError("");
    }
    if (rerender) renderEvents();
    else {
      updateEventSummary(event);
      loadSelectedMedia(event);
    }
  }

  async function loadSelectedMedia(event) {
    downloadGeneration += 1;
    downloadEvent = event;
    downloadButton.disabled = !event.clip_url;
    downloadButton.textContent = "↓ 下载视频";
    downloadButton.title = "保存仅带人物框和手机框的 MP4 视频";
    downloadButton.setAttribute("aria-busy", "false");
    const generation = selectionGate.begin();
    if (overlayController) overlayController.abort();
    overlayController = new AbortController();
    selectedMediaKey = eventMediaKey(event);
    overlayTimeline = [];
    renderEvidenceChain(event, overlayTimeline);
    clearCanvas();
    emptyState.hidden = true;
    if (video.getAttribute("src") !== event.clip_url) {
      video.src = event.clip_url;
      video.load();
    }
    try {
      const payload = await fetchJson(event.overlay_url, overlayController.signal);
      if (!selectionGate.isCurrent(generation) || dashboardState.selectedId !== event.event_id) return;
      const timeline = payload.overlay && payload.overlay.bbox_timeline;
      overlayTimeline = Array.isArray(timeline) ? timeline : [];
      renderEvidenceChain(event, overlayTimeline);
      drawOverlay();
    } catch (error) {
      if (error.name !== "AbortError" && selectionGate.isCurrent(generation)) {
        overlayTimeline = [];
        renderEvidenceChain(event, overlayTimeline);
        clearCanvas();
      }
    }
  }

  function updateEventSummary(event) {
    setText("event-camera", cameraLabel(eventCamera(event)));
    renderEvidenceChain(event, overlayTimeline);
    renderReviewControls(event);
  }

  function renderEvidenceChain(event, timeline) {
    const evidence = decisionEvidence(event, timeline);
    const statusText = evidence.hasDetailedEvidence
      ? "完整证据"
      : timeline.length
        ? "基础证据"
        : "读取证据中";
    const statusClass = `evidence-chain-status ${evidence.isAlarm ? "is-alarm" : ""}`;
    const evidenceKey = JSON.stringify([
      statusText,
      statusClass,
      evidence.steps.map((step) => [step.state, step.label, step.value, step.detail]),
    ]);
    if (evidenceKey === renderedEvidenceKey) return;
    renderedEvidenceKey = evidenceKey;
    evidenceChainStatus.textContent = statusText;
    evidenceChainStatus.className = statusClass;
    evidenceChainList.replaceChildren(...evidence.steps.map((step) => {
      const card = document.createElement("div");
      card.className = `evidence-step is-${step.state}`;
      const label = document.createElement("span");
      label.textContent = step.label;
      const value = document.createElement("strong");
      value.textContent = step.value;
      const detail = document.createElement("small");
      detail.textContent = step.detail;
      card.append(label, value, detail);
      return card;
    }));
  }

  function renderReviewControls(event) {
    const result = reviewResult(event);
    const category = reviewCategory(event);
    const saving = reviewSavePending(pendingReviews, event);
    const disabled = !event || saving;
    const controlsKey = JSON.stringify([result, category, saving, disabled]);
    if (controlsKey === renderedReviewControlsKey) return;
    renderedReviewControlsKey = controlsKey;
    reviewActions.setAttribute("aria-busy", String(saving));
    reviewControls.forEach((button) => {
      const selected = button.dataset.reviewResult === result
        && (result !== "confirmed" || button.dataset.reviewCategory === category);
      button.disabled = disabled;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
  }

  function setReviewError(message) {
    reviewError.textContent = message;
    reviewError.hidden = !message;
  }

  function hideReviewToast() {
    if (reviewToastTimer !== null) clearTimeout(reviewToastTimer);
    reviewToastTimer = null;
    undoReview = null;
    reviewToast.hidden = true;
  }

  function showFalsePositiveHiddenToast(event, previousResult, previousReason, previousCategory) {
    hideReviewToast();
    undoReview = { event, previousResult, previousReason, previousCategory };
    reviewToastMessage.textContent = "已标记误报并从事件列表隐藏";
    reviewToast.hidden = false;
    reviewToastTimer = setTimeout(hideReviewToast, 8000);
  }

  async function undoHiddenFalsePositive() {
    if (!undoReview) return;
    const undo = undoReview;
    hideReviewToast();
    reviewToastUndo.disabled = true;
    try {
      const request = createReviewRequest(
        undo.event,
        undo.previousResult,
        undo.previousReason,
        undo.previousCategory,
      );
      const response = await fetch(request.url, request.options);
      if (!response.ok) throw await reviewRequestError(response);
      const payload = await response.json();
      if (!REVIEW_RESULTS.has(payload.review_result)) throw new Error("invalid response");
      reviewOverrides.set(undo.event.event_id, {
        result: payload.review_result,
        category: payload.review_category || null,
      });
      eventPoller.refresh();
    } catch (_error) {
      setReviewError("撤销失败，可开启“显示已标误报”后重新修改");
    } finally {
      reviewToastUndo.disabled = false;
    }
  }

  async function submitReview(result, reason = null, category = null) {
    const selected = events.find((event) => event.event_id === dashboardState.selectedId);
    if (!selected || reviewSavePending(pendingReviews, selected) || !REVIEW_RESULTS.has(result)) {
      return;
    }
    if (
      result === "false_positive"
      && (reason === null || !FALSE_POSITIVE_REASONS.has(reason))
    ) {
      falsePositiveDialog.showModal();
      return;
    }
    const eventId = selected.event_id;
    const previousResult = reviewResult(selected);
    const previousReason = selected.review_reason || null;
    const previousCategory = reviewCategory(selected) || null;
    const previousOverride = reviewOverrides.get(eventId);
    pendingReviews.set(eventId, { previousResult, result, reason, category });
    reviewOverrides.set(eventId, { result, category });
    setReviewError("");
    events = updateEventReview(events, eventId, result, reason, category);
    renderEvents();
    try {
      const request = createReviewRequest(selected, result, reason, category);
      const response = await fetch(request.url, request.options);
      if (!response.ok) throw await reviewRequestError(response);
      const payload = await response.json();
      if (!REVIEW_RESULTS.has(payload.review_result)) throw new Error("invalid response");
      if (reason === "fixed_phone" && payload.fixed_template_created !== true) {
        if (dashboardState.selectedId === eventId) {
          setReviewError("误报已保存，但当前事件不足3帧有效手机框，未生成固定手机模板");
        }
      }
      reviewOverrides.set(eventId, {
        result: payload.review_result,
        category: payload.review_category || null,
      });
      events = updateEventReview(
        events,
        eventId,
        payload.review_result,
        payload.review_reason || null,
        payload.review_category || null
      );
      if (
        payload.review_result === "false_positive"
        && !dashboardState.showFalsePositives
      ) {
        events = events.filter((event) => event.event_id !== eventId);
        showFalsePositiveHiddenToast(selected, previousResult, previousReason, previousCategory);
      }
      // Discard a request started before the save. The asynchronous server
      // index may still return older rows; reviewAdjustedSummary bridges that.
      refreshGate.begin();
      if (eventRequestController) eventRequestController.abort();
      eventPoller.refresh();
    } catch (error) {
      if (previousOverride) reviewOverrides.set(eventId, previousOverride);
      else reviewOverrides.delete(eventId);
      events = updateEventReview(events, eventId, previousResult, previousReason, previousCategory);
      if (dashboardState.selectedId === eventId) {
        setReviewError(reviewSaveErrorMessage(error));
      }
    } finally {
      pendingReviews.delete(eventId);
      renderEvents();
    }
  }

  function renderFixedObjects(templates) {
    fixedObjectList.replaceChildren();
    const records = Array.isArray(templates) ? templates : [];
    fixedObjectEmpty.hidden = records.length > 0;
    records.forEach((template) => {
      const card = document.createElement("article");
      card.className = "fixed-object-card";
      const image = document.createElement("img");
      image.alt = `${template.camera || "未知视角"}固定手机模板`;
      if (template.thumbnail_url) image.src = template.thumbnail_url;
      const details = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = template.camera || "未知视角";
      const metadata = document.createElement("span");
      metadata.textContent = `${formatTimestamp(template.occurred_at)} · ${template.sample_count || 0}张`;
      details.append(title, metadata);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "删除";
      remove.addEventListener("click", async () => {
        remove.disabled = true;
        fixedObjectError.hidden = true;
        try {
          const response = await fetch(
            `/api/fixed-objects/${encodeURIComponent(template.template_id)}`,
            { method: "DELETE" }
          );
          if (!response.ok) throw await reviewRequestError(response);
          card.remove();
          fixedObjectEmpty.hidden = fixedObjectList.childElementCount > 0;
        } catch (_error) {
          remove.disabled = false;
          fixedObjectError.textContent = "固定手机模板删除失败，请重试";
          fixedObjectError.hidden = false;
        }
      });
      card.append(image, details, remove);
      fixedObjectList.append(card);
    });
  }

  async function openFixedObjectManager() {
    fixedObjectError.hidden = true;
    fixedObjectDialog.showModal();
    try {
      const payload = await fetchJson("/api/fixed-objects");
      renderFixedObjects(payload.templates);
    } catch (_error) {
      fixedObjectError.textContent = "固定手机模板加载失败，请重试";
      fixedObjectError.hidden = false;
    }
  }

  function cloneScreenList(value) {
    return Array.isArray(value)
      ? value.map((screen, index) => ({
          screen_id: String((screen && screen.screen_id) || `screen_${String(index + 1).padStart(2, "0")}`),
          screen_poly: Array.isArray(screen && screen.screen_poly)
            ? screen.screen_poly.map((point) => [Number(point[0]), Number(point[1])])
            : [],
        }))
      : [];
  }

  function cameraManagementMessage(message, kind = "") {
    cameraManagementStatus.textContent = message || "";
    cameraManagementStatus.className = `camera-management-status${kind ? ` is-${kind}` : ""}`;
  }

  function cameraManagementRuntime(camera) {
    const relay = String(camera && camera.relay || "");
    return cameras.find((item) => String(item && item.relay || "") === relay) || null;
  }

  function cameraManagementCalibration(camera) {
    const relay = String(camera && camera.relay || "");
    const edited = screenCalibrationEdits.get(relay);
    const existing = cameraManagement && cameraManagement.calibrations && cameraManagement.calibrations[relay];
    const screens = edited || (existing && existing.screens) || [];
    const count = Array.isArray(screens) ? screens.length : 0;
    if (camera && camera.enabled === false) {
      return { key: "disabled", label: "未启用", detail: "不会参与当前推理", count };
    }
    if (camera && camera.has_screen === false) {
      return { key: "not_required", label: "无需标定", detail: "该视角不监测屏幕", count: 0 };
    }
    if (!count) {
      return { key: "needs_calibration", label: "待标定", detail: "尚未设置屏幕位置", count: 0 };
    }
    return {
      key: "complete",
      label: edited ? "草稿已标定" : "已标定",
      detail: `已标定 ${count} 块屏幕`,
      count,
    };
  }

  function cameraManagementOnlineLabel(camera) {
    const runtime = cameraManagementRuntime(camera);
    if (camera && camera.enabled === false) return "未启用";
    if (!runtime) return "等待运行状态";
    if (runtime.status === "online") {
      const fps = Number(runtime.fps);
      return Number.isFinite(fps) ? `在线 · ${Math.round(fps)} FPS` : "在线";
    }
    return cameraHealthText(runtime, false);
  }

  function cameraFromManagementCard(card) {
    const relay = cardCameraValue(card, "relay");
    const original = (cameraManagement && cameraManagement.cameras || [])
      .find((camera) => String(camera && camera.relay || "") === relay) || {};
    return {
      ...original,
      relay,
      view: cardCameraValue(card, "view"),
      enabled: Boolean(cardCameraValue(card, "enabled")),
      has_screen: Boolean(cardCameraValue(card, "has_screen")),
    };
  }

  function applyCameraManagementCardState(card) {
    const camera = cameraFromManagementCard(card);
    const calibration = cameraManagementCalibration(camera);
    const enabled = camera.enabled !== false;
    card.dataset.calibrationState = calibration.key;
    card.classList.toggle("is-disabled", !enabled);
    const status = card.querySelector(".camera-management-calibration-state");
    const detail = card.querySelector(".camera-management-calibration-detail");
    const runtime = card.querySelector(".camera-management-runtime-state");
    if (status) {
      status.textContent = calibration.label;
      status.className = `camera-management-calibration-state is-${calibration.key}`;
    }
    if (detail) detail.textContent = calibration.detail;
    if (runtime) runtime.textContent = cameraManagementOnlineLabel(camera);
  }

  function refreshCameraManagementOverview() {
    const cards = [...cameraManagementList.querySelectorAll(".camera-management-card")];
    const summary = { required: 0, complete: 0, pending: 0, disabled: 0, notRequired: 0 };
    cards.forEach((card) => {
      applyCameraManagementCardState(card);
      const state = card.dataset.calibrationState;
      if (state === "disabled") summary.disabled += 1;
      else if (state === "not_required") summary.notRequired += 1;
      else {
        summary.required += 1;
        if (state === "complete") summary.complete += 1;
        else summary.pending += 1;
      }
      card.hidden = cameraManagementFilter !== "all" && state !== cameraManagementFilter;
    });
    cameraManagementSummary.replaceChildren();
    const progress = document.createElement("strong");
    progress.textContent = `屏幕标定 ${summary.complete}/${summary.required}`;
    const detail = document.createElement("span");
    detail.textContent = summary.pending
      ? `还有 ${summary.pending} 路待标定${summary.disabled ? `；${summary.disabled} 路未启用` : ""}`
      : `需要监测的视角均已标定${summary.disabled ? `；${summary.disabled} 路未启用` : ""}`;
    cameraManagementSummary.append(progress, detail);
    cameraManagementFilters.querySelectorAll("[data-camera-management-filter]").forEach((button) => {
      const selected = button.dataset.cameraManagementFilter === cameraManagementFilter;
      button.setAttribute("aria-pressed", String(selected));
    });
  }

  function clearCameraManagementPreviews() {
    cameraManagementPreviewRequest += 1;
    cameraManagementPreviewUrls.forEach((url) => URL.revokeObjectURL(url));
    cameraManagementPreviewUrls.clear();
  }

  function updateCameraManagementPreview(card, state, message = "") {
    const frame = card.querySelector(".camera-management-preview");
    const image = card.querySelector(".camera-management-preview img");
    const label = card.querySelector(".camera-management-preview-label");
    if (!frame || !image || !label) return;
    frame.classList.toggle("is-loading", state === "loading");
    frame.classList.toggle("is-error", state === "error");
    frame.classList.toggle("is-disabled", state === "disabled");
    if (state !== "ready") image.removeAttribute("src");
    label.textContent = message;
  }

  async function loadCameraManagementPreview(card, requestId) {
    const relay = cardCameraValue(card, "relay");
    const camera = cameraFromManagementCard(card);
    if (!relay || camera.enabled === false) {
      updateCameraManagementPreview(card, "disabled", "未启用，不读取画面");
      return;
    }
    const cached = cameraManagementPreviewUrls.get(relay);
    if (cached) {
      const image = card.querySelector(".camera-management-preview img");
      if (image) image.src = cached;
      updateCameraManagementPreview(card, "ready", "当前标定背景");
      return;
    }
    updateCameraManagementPreview(card, "loading", "正在读取标定背景…");
    try {
      const response = await fetch(`/api/camera-management/${encodeURIComponent(relay)}/preview?t=${Date.now()}`, {
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const image = await response.blob();
      if (!image.size || !String(image.type || "").startsWith("image/")) {
        throw new Error("服务器未返回有效图片");
      }
      const objectUrl = URL.createObjectURL(image);
      if (requestId !== cameraManagementPreviewRequest || !cameraManagementDialog.open || !card.isConnected) {
        URL.revokeObjectURL(objectUrl);
        return;
      }
      cameraManagementPreviewUrls.set(relay, objectUrl);
      const preview = card.querySelector(".camera-management-preview img");
      if (preview) preview.src = objectUrl;
      updateCameraManagementPreview(card, "ready", "当前标定背景");
    } catch (_error) {
      if (requestId !== cameraManagementPreviewRequest || !cameraManagementDialog.open || !card.isConnected) return;
      updateCameraManagementPreview(card, "error", "当前背景读取失败");
    }
  }

  function loadCameraManagementPreviews() {
    const requestId = ++cameraManagementPreviewRequest;
    const cards = [...cameraManagementList.querySelectorAll(".camera-management-card")];
    let index = 0;
    const worker = async () => {
      while (index < cards.length) {
        const card = cards[index];
        index += 1;
        await loadCameraManagementPreview(card, requestId);
      }
    };
    // Fill cards one by one.  A gallery is a manual configuration surface, so
    // reliable key-frame capture matters more than shaving a few seconds from
    // its initial fill; this avoids competing HEVC reader startups.
    void worker();
  }

  function managementCameraCard(camera, index) {
    const card = document.createElement("article");
    card.className = `camera-management-card${camera.enabled === false ? " is-disabled" : ""}`;
    card.dataset.cameraRelay = camera.relay || "";
    const preview = document.createElement("div");
    preview.className = "camera-management-preview is-loading";
    const previewImage = document.createElement("img");
    previewImage.alt = `${camera.view || camera.relay || `第${index + 1}路摄像头`}当前标定背景`;
    const previewLabel = document.createElement("span");
    previewLabel.className = "camera-management-preview-label";
    previewLabel.textContent = "等待读取当前背景…";
    preview.append(previewImage, previewLabel);

    const header = document.createElement("div");
    header.className = "camera-management-card-header";
    const identity = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = camera.view || camera.relay || `第${index + 1}路摄像头`;
    const runtime = document.createElement("small");
    runtime.className = "camera-management-runtime-state";
    identity.append(title, runtime);
    const enabled = document.createElement("label");
    enabled.className = "camera-enabled-toggle";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = camera.enabled !== false;
    toggle.dataset.cameraField = "enabled";
    toggle.addEventListener("change", () => refreshCameraManagementOverview());
    enabled.append(toggle, document.createTextNode("启用"));
    header.append(identity, enabled);

    const calibration = document.createElement("div");
    calibration.className = "camera-management-calibration";
    const calibrationState = document.createElement("strong");
    calibrationState.className = "camera-management-calibration-state";
    const calibrationDetail = document.createElement("span");
    calibrationDetail.className = "camera-management-calibration-detail";
    calibration.append(calibrationState, calibrationDetail);

    const quickActions = document.createElement("div");
    quickActions.className = "camera-management-quick-actions";
    const calibrate = document.createElement("button");
    calibrate.type = "button";
    calibrate.className = "primary";
    calibrate.textContent = "标定屏幕";
    calibrate.addEventListener("click", () => openScreenCalibration(card));
    const settings = document.createElement("button");
    settings.type = "button";
    settings.textContent = "连接设置";
    quickActions.append(calibrate, settings);

    const fields = document.createElement("div");
    fields.className = "camera-management-fields";
    const relay = document.createElement("input");
    relay.type = "hidden";
    relay.value = camera.relay || "";
    relay.dataset.cameraField = "relay";
    card.append(relay);
    const addField = (label, field, value, options = {}) => {
      const wrapper = document.createElement("label");
      wrapper.textContent = label;
      const input = document.createElement("input");
      input.type = options.type || "text";
      input.value = value === undefined || value === null ? "" : String(value);
      input.placeholder = options.placeholder || "";
      input.autocomplete = options.autocomplete || "off";
      input.dataset.cameraField = field;
      if (options.readOnly) input.readOnly = true;
      wrapper.append(input);
      (options.container || fields).append(wrapper);
    };
    addField("显示名称", "view", camera.view, { placeholder: "例如 FFS-北侧" });
    addField("NVR 地址", "host", camera.host, { placeholder: "192.168.x.x" });
    addField("NVR 通道号", "channel", nvrChannelFromPath(camera.path), { placeholder: "例如 D4、38" });
    addField("用户名", "username", camera.username, { autocomplete: "username" });
    addField("密码", "password", "", { type: "password", placeholder: camera.password_configured ? "保持原密码" : "新摄像头必填", autocomplete: "new-password" });

    const advanced = document.createElement("details");
    advanced.className = "camera-management-advanced";
    const advancedSummary = document.createElement("summary");
    advancedSummary.textContent = "高级连接设置（一般无需修改）";
    const advancedFields = document.createElement("div");
    advancedFields.className = "camera-management-fields";
    addField("RTSP 端口", "port", camera.port || 554, {
      type: "number", container: advancedFields, placeholder: "默认 554",
    });
    addField("RTSP 路径", "path", camera.path || "/Streaming/Channels/101", {
      container: advancedFields, placeholder: "非标准 NVR 才需要填写",
    });
    advanced.append(advancedSummary, advancedFields);

    const secondary = document.createElement("div");
    secondary.className = "camera-management-secondary";
    const hasScreen = document.createElement("label");
    const screenToggle = document.createElement("input");
    screenToggle.type = "checkbox";
    screenToggle.checked = camera.has_screen !== false;
    screenToggle.dataset.cameraField = "has_screen";
    screenToggle.addEventListener("change", () => refreshCameraManagementOverview());
    hasScreen.append(screenToggle, document.createTextNode("该画面包含需监测的屏幕"));
    const buttons = document.createElement("div");
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "删除摄像头";
    remove.addEventListener("click", () => {
      card.remove();
      cameraManagementMessage("已从待发布配置移除；保存后才生效", "pending");
      refreshCameraManagementOverview();
    });
    buttons.append(remove);
    secondary.append(hasScreen, buttons);
    const settingsPanel = document.createElement("div");
    settingsPanel.className = "camera-management-settings";
    settingsPanel.hidden = true;
    settings.addEventListener("click", () => {
      const expanded = settingsPanel.hidden;
      settingsPanel.hidden = !expanded;
      settings.textContent = expanded ? "收起设置" : "连接设置";
    });
    settingsPanel.append(fields, advanced, secondary);
    card.append(preview, header, calibration, quickActions, settingsPanel);
    applyCameraManagementCardState(card);
    return card;
  }

  function renderCameraManagement() {
    const state = cameraManagement || {};
    const records = Array.isArray(state.cameras) ? state.cameras : [];
    cameraManagementList.replaceChildren(...records.map(managementCameraCard));
    const limit = Number(state.max_active_cameras) || 15;
    const active = Number(state.active_cameras) || 0;
    const draftActive = Number(state.draft_active_cameras) || active;
    const pending = state.pending_activation === true;
    cameraManagementDiscard.disabled = !pending || cameraManagementBusy;
    cameraManagementSave.disabled = cameraManagementBusy;
    cameraManagementAdd.disabled = cameraManagementBusy;
    cameraManagementPublish.disabled = !pending || cameraManagementBusy;
    cameraManagementMessage(
      pending
        ? `已有待发布草稿：${draftActive}/${limit} 路计划启用；当前线上仍为 ${active}/${limit} 路。`
        : `当前线上启用 ${active}/${limit} 路。保存后只形成待发布草稿，不会打断线上推理。`,
      pending ? "pending" : "",
    );
    refreshCameraManagementOverview();
    loadCameraManagementPreviews();
  }

  async function openCameraManagement() {
    cameraManagementDialog.showModal();
    cameraManagementMessage("正在读取摄像头配置…");
    try {
      cameraManagement = await fetchJson("/api/camera-management");
      screenCalibrationEdits.clear();
      clearCameraManagementPreviews();
      cameraManagementFilter = "all";
      renderCameraManagement();
    } catch (_error) {
      cameraManagementMessage("摄像头配置读取失败，请确认当前服务版本已启用此功能", "error");
    }
  }

  function cardCameraValue(card, field) {
    const node = card.querySelector(`[data-camera-field="${field}"]`);
    if (!node) return "";
    return node.type === "checkbox" ? node.checked : node.value.trim();
  }

  function buildCameraManagementDraft() {
    if (!cameraManagement) throw new Error("未加载摄像头配置");
    const existing = new Map((cameraManagement.cameras || []).map((camera) => [camera.relay, camera]));
    const nextCameras = [...cameraManagementList.querySelectorAll(".camera-management-card")].map((card) => {
      const relay = cardCameraValue(card, "relay");
      const original = existing.get(relay);
      const channel = cardCameraValue(card, "channel");
      let path = cardCameraValue(card, "path");
      if (channel) {
        path = nvrRtspPathFromChannel(channel);
        if (!path) {
          const label = cardCameraValue(card, "view") || relay || "该摄像头";
          throw new Error(`${label} 的 NVR 通道号无效，请填写如 D4、38`);
        }
      }
      const camera = {
        relay,
        view: cardCameraValue(card, "view"),
        host: cardCameraValue(card, "host"),
        port: Number(cardCameraValue(card, "port") || 554),
        path,
        channel,
        username: cardCameraValue(card, "username"),
        password: cardCameraValue(card, "password"),
        calibration: original && original.calibration || normalizedCalibrationName(relay),
        enabled: Boolean(cardCameraValue(card, "enabled")),
        has_screen: Boolean(cardCameraValue(card, "has_screen")),
      };
      if (!camera.password && original) delete camera.password;
      return camera;
    });
    const calibrationMap = {};
    nextCameras.forEach((camera) => {
      const edited = screenCalibrationEdits.get(camera.relay);
      const existingCalibration = cameraManagement.calibrations && cameraManagement.calibrations[camera.relay];
      if (edited) calibrationMap[camera.relay] = { screens: cloneScreenList(edited) };
      else if (existingCalibration) calibrationMap[camera.relay] = { screens: cloneScreenList(existingCalibration.screens) };
      else if (camera.has_screen) calibrationMap[camera.relay] = { screens: [] };
    });
    return {
      runtime: cameraManagement.runtime || {},
      cameras: nextCameras,
      calibrations: calibrationMap,
    };
  }

  async function saveCameraManagementDraft() {
    if (cameraManagementBusy) return;
    let draft;
    let saved = false;
    try {
      draft = buildCameraManagementDraft();
    } catch (error) {
      cameraManagementMessage(error.message || "配置整理失败", "error");
      return;
    }
    cameraManagementBusy = true;
    cameraManagementSave.disabled = true;
    cameraManagementAdd.disabled = true;
    cameraManagementMessage("正在校验并保存待发布配置…");
    try {
      const response = await fetch("/api/camera-management/draft", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft),
      });
      if (!response.ok) throw await reviewRequestError(response);
      cameraManagement = await response.json();
      screenCalibrationEdits.clear();
      saved = true;
      renderCameraManagement();
    } catch (error) {
      // Keep the operator's current form in place.  In particular, a newly
      // added camera may fail validation before its first screen calibration;
      // replacing the card list here would silently discard all connection
      // fields the operator just entered.
      cameraManagementMessage(error.message || "保存失败，请检查必填项和屏幕标定", "error");
    } finally {
      cameraManagementBusy = false;
      if (saved && cameraManagement) {
        renderCameraManagement();
      } else if (cameraManagement && cameraManagementSave.isConnected) {
        cameraManagementSave.disabled = false;
        cameraManagementAdd.disabled = false;
      }
    }
  }

  async function discardCameraManagementDraft() {
    if (cameraManagementBusy || !cameraManagement || cameraManagement.pending_activation !== true) return;
    if (!window.confirm("放弃待发布摄像头配置？线上正在运行的配置不会受到影响。")) return;
    cameraManagementBusy = true;
    renderCameraManagement();
    try {
      const response = await fetch("/api/camera-management/draft", { method: "DELETE" });
      if (!response.ok) throw await reviewRequestError(response);
      cameraManagement = await response.json();
      screenCalibrationEdits.clear();
      renderCameraManagement();
    } catch (error) {
      cameraManagementMessage(error.message || "放弃草稿失败", "error");
    } finally {
      cameraManagementBusy = false;
      if (cameraManagement) renderCameraManagement();
    }
  }

  async function publishCameraManagementDraft() {
    if (cameraManagementBusy || !cameraManagement || cameraManagement.pending_activation !== true) return;
    if (!window.confirm("确认应用待发布配置？系统会先预检摄像头连接，再自动重启实时推理。")) return;
    cameraManagementBusy = true;
    renderCameraManagement();
    cameraManagementMessage("正在预检并发布配置，页面将短暂重新连接…", "pending");
    try {
      const response = await fetch("/api/camera-management/publish", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) throw await reviewRequestError(response);
      const result = await response.json();
      cameraManagementMessage(result.message || "已开始发布，请等待系统恢复运行。", "pending");
      cameraManagementRestarting = true;
      void waitForCameraManagementRestart();
    } catch (error) {
      cameraManagementMessage(error.message || "发布未启动，请检查待发布配置", "error");
      cameraManagementBusy = false;
      if (cameraManagement) renderCameraManagement();
    }
  }

  async function waitForCameraManagementRestart() {
    const startedAt = Date.now();
    const deadline = startedAt + 90000;
    let sawDisconnect = false;
    // Do not reload a still-running old dashboard after an arbitrary five
    // seconds. Wait for the intentional disconnect, then reload only after
    // the replacement dashboard has answered again.
    await new Promise((resolve) => window.setTimeout(resolve, 8000));
    while (Date.now() < deadline) {
      try {
        const response = await fetch("/api/status", { cache: "no-store" });
        // A normal restart produces a brief disconnect.  On an unusually
        // fast host the browser can miss that interval, so accept a healthy
        // replacement after a conservative grace period as well.
        if (response.ok && (sawDisconnect || Date.now() - startedAt >= 30000)) {
          window.location.reload();
          return;
        }
      } catch (_error) {
        sawDisconnect = true;
      }
      const elapsed = Math.floor((Date.now() - startedAt) / 1000);
      cameraManagementMessage(`配置已发布，正在安全重启（${elapsed}s）…`, "pending");
      await new Promise((resolve) => window.setTimeout(resolve, 1000));
    }
    cameraManagementRestarting = false;
    cameraManagementBusy = false;
    cameraManagementMessage("重启超过90秒仍未恢复；请刷新页面后查看运行状态。", "error");
    if (cameraManagement) renderCameraManagement();
  }

  function addCameraManagementCamera() {
    const used = new Set(
      [...cameraManagementList.querySelectorAll(".camera-management-card")]
        .map((card) => cardCameraValue(card, "relay"))
        .filter(Boolean),
    );
    let number = 1;
    while (used.has(`camera${String(number).padStart(2, "0")}`)) number += 1;
    const relay = `camera${String(number).padStart(2, "0")}`;
    const count = cameraManagementList.childElementCount + 1;
    const card = managementCameraCard({
      relay,
      view: `新摄像头${count}`,
      host: "",
      port: 554,
      path: "/Streaming/Channels/101",
      calibration: normalizedCalibrationName(relay),
      enabled: false,
      // A new stream must first be published so its real background can be
      // captured.  Do not require a non-existent calibration before that.
      has_screen: false,
      username: "",
      password_configured: false,
    }, count - 1);
    cameraManagementList.append(card);
    cameraManagementMessage("已新增未启用摄像头；填写连接信息和屏幕位置后保存草稿。", "pending");
    refreshCameraManagementOverview();
    loadCameraManagementPreviews();
  }

  function currentCalibrationCanvasScale() {
    if (!screenCalibration) return null;
    const rect = screenCalibrationCanvas.getBoundingClientRect();
    const width = Number(screenCalibration.frameSize[0]) || 2560;
    const height = Number(screenCalibration.frameSize[1]) || 1440;
    if (!rect.width || !rect.height) return null;
    return { x: screenCalibrationCanvas.width / width, y: screenCalibrationCanvas.height / height, rect, width, height };
  }

  function resizeCalibrationCanvas() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, screenCalibrationStage.clientWidth);
    const height = Math.max(1, screenCalibrationStage.clientHeight);
    screenCalibrationCanvas.width = Math.round(width * ratio);
    screenCalibrationCanvas.height = Math.round(height * ratio);
    drawScreenCalibration();
  }

  function activeCalibrationScreen() {
    if (!screenCalibration || !screenCalibration.screens.length) return null;
    return screenCalibration.screens[Math.min(screenCalibration.selected, screenCalibration.screens.length - 1)] || null;
  }

  function renderScreenCalibrationSelect() {
    const selected = screenCalibration ? screenCalibration.selected : 0;
    const items = screenCalibration ? screenCalibration.screens : [];
    screenCalibrationSelect.replaceChildren(...items.map((screen, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = screen.screen_id || `屏幕 ${index + 1}`;
      option.selected = index === selected;
      return option;
    }));
    screenCalibrationRemove.disabled = items.length <= 1;
  }

  function drawScreenCalibration() {
    const context = screenCalibrationCanvas.getContext("2d");
    context.clearRect(0, 0, screenCalibrationCanvas.width, screenCalibrationCanvas.height);
    const scale = currentCalibrationCanvasScale();
    if (!scale || !screenCalibration) return;
    screenCalibration.screens.forEach((screen, index) => {
      const active = index === screenCalibration.selected;
      const points = screen.screen_poly || [];
      if (points.length < 2) return;
      context.beginPath();
      points.forEach((point, pointIndex) => {
        const x = point[0] * scale.x;
        const y = point[1] * scale.y;
        if (pointIndex === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.closePath();
      context.fillStyle = active ? "rgba(229, 62, 62, .18)" : "rgba(22, 119, 232, .13)";
      context.strokeStyle = active ? "#ee3d36" : "#247bc5";
      context.lineWidth = active ? 3 : 2;
      context.fill();
      context.stroke();
      if (!active) return;
      points.forEach((point, pointIndex) => {
        const x = point[0] * scale.x;
        const y = point[1] * scale.y;
        context.beginPath();
        context.arc(x, y, Math.max(5, screenCalibrationCanvas.width / 150), 0, Math.PI * 2);
        context.fillStyle = "#fff";
        context.fill();
        context.lineWidth = 3;
        context.strokeStyle = "#ee3d36";
        context.stroke();
        context.fillStyle = "#a61d19";
        context.font = `${Math.max(11, screenCalibrationCanvas.width / 75)}px sans-serif`;
        context.fillText(String(pointIndex + 1), x + 8, y - 8);
      });
    });
  }

  function defaultScreenPolygon(frameSize) {
    const width = Number(frameSize[0]) || 2560;
    const height = Number(frameSize[1]) || 1440;
    return [
      [width * .36, height * .35], [width * .64, height * .35],
      [width * .64, height * .65], [width * .36, height * .65],
    ];
  }

  function stashScreenCalibrationEdit() {
    if (!screenCalibration) return false;
    const invalid = screenCalibration.screens.some((screen) => !Array.isArray(screen.screen_poly) || screen.screen_poly.length < 3);
    if (invalid) {
      screenCalibrationError.textContent = "每个屏幕至少需要 3 个顶点";
      screenCalibrationError.hidden = false;
      return false;
    }
    screenCalibrationEdits.set(screenCalibration.relay, cloneScreenList(screenCalibration.screens));
    refreshCameraManagementOverview();
    return true;
  }

  async function openScreenCalibration(card) {
    const relay = cardCameraValue(card, "relay");
    if (!relay) {
      cameraManagementMessage("请先保存摄像头基本信息，再编辑屏幕位置", "error");
      return;
    }
    const current = cameraManagement && cameraManagement.calibrations && cameraManagement.calibrations[relay];
    const frameSize = (current && current.frame_size) || (cameraManagement && [cameraManagement.runtime.source_width, cameraManagement.runtime.source_height]) || [2560, 1440];
    const sourceScreens = screenCalibrationEdits.get(relay) || (current && current.screens) || [];
    screenCalibration = {
      relay,
      frameSize,
      original: cloneScreenList(sourceScreens),
      screens: cloneScreenList(sourceScreens),
      selected: 0,
    };
    if (!screenCalibration.screens.length) {
      screenCalibration.screens.push({ screen_id: "screen_01", screen_poly: defaultScreenPolygon(frameSize) });
    }
    screenCalibrationCamera.textContent = relay;
    screenCalibrationError.hidden = true;
    renderScreenCalibrationSelect();
    screenCalibrationImage.removeAttribute("src");
    if (screenCalibrationPreviewObjectUrl) {
      URL.revokeObjectURL(screenCalibrationPreviewObjectUrl);
      screenCalibrationPreviewObjectUrl = "";
    }
    const requestId = ++screenCalibrationPreviewRequest;
    screenCalibrationLoading.textContent = "正在读取当前画面…";
    screenCalibrationLoading.hidden = false;
    if (!screenCalibrationDialog.open) screenCalibrationDialog.showModal();
    resizeCalibrationCanvas();
    const previewUrl = `/api/camera-management/${encodeURIComponent(relay)}/preview?t=${Date.now()}`;
    try {
      // Fetch the preview first instead of relying on the browser's implicit
      // <img> request. This makes non-200 responses observable and, together
      // with requestId, prevents an old failed request from leaving the
      // loading mask above a newer successful camera frame.
      const response = await fetch(previewUrl, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const image = await response.blob();
      if (requestId !== screenCalibrationPreviewRequest || !screenCalibrationDialog.open) return;
      if (!image.size || !String(image.type || "").startsWith("image/")) {
        throw new Error("服务器未返回有效图片");
      }
      const objectUrl = URL.createObjectURL(image);
      const decoder = new Image();
      await new Promise((resolve, reject) => {
        decoder.onload = () => resolve();
        decoder.onerror = () => reject(new Error("浏览器无法解码标定图片"));
        decoder.src = objectUrl;
      });
      if (requestId !== screenCalibrationPreviewRequest || !screenCalibrationDialog.open) {
        URL.revokeObjectURL(objectUrl);
        return;
      }
      screenCalibrationPreviewObjectUrl = objectUrl;
      screenCalibrationImage.src = objectUrl;
      screenCalibrationLoading.hidden = true;
      window.requestAnimationFrame(resizeCalibrationCanvas);
    } catch (error) {
      if (requestId !== screenCalibrationPreviewRequest || !screenCalibrationDialog.open) return;
      const message = error instanceof Error ? error.message : "未知错误";
      screenCalibrationLoading.textContent = `当前画面获取失败（${message}）；仍可在坐标画布上修改，保存前请复核。`;
      window.requestAnimationFrame(resizeCalibrationCanvas);
    }
  }

  function calibrationPointFromEvent(event) {
    const scale = currentCalibrationCanvasScale();
    if (!scale) return null;
    const x = (event.clientX - scale.rect.left) * screenCalibrationCanvas.width / scale.rect.width / scale.x;
    const y = (event.clientY - scale.rect.top) * screenCalibrationCanvas.height / scale.rect.height / scale.y;
    return [Math.max(0, Math.min(scale.width, x)), Math.max(0, Math.min(scale.height, y))];
  }

  function nearestCalibrationVertex(event) {
    const point = calibrationPointFromEvent(event);
    const screen = activeCalibrationScreen();
    const scale = currentCalibrationCanvasScale();
    if (!point || !screen || !scale) return null;
    const threshold = 18 / Math.min(scale.x, scale.y);
    let nearest = null;
    (screen.screen_poly || []).forEach((candidate, index) => {
      const distance = Math.hypot(candidate[0] - point[0], candidate[1] - point[1]);
      if (distance <= threshold && (!nearest || distance < nearest.distance)) nearest = { index, distance };
    });
    return nearest;
  }

  function saveScreenCalibrationEdit() {
    if (!stashScreenCalibrationEdit()) return;
    screenCalibrationDialog.close();
    cameraManagementMessage(`${screenCalibration.relay} 的屏幕位置已保存到本次草稿，点击“保存待发布配置”后写入。`, "pending");
  }

  function resetReview() {
    downloadGeneration += 1;
    downloadEvent = null;
    downloadButton.disabled = true;
    downloadButton.textContent = "↓ 下载视频";
    downloadButton.setAttribute("aria-busy", "false");
    selectionGate.begin();
    if (overlayController) overlayController.abort();
    video.removeAttribute("src");
    video.load();
    overlayTimeline = [];
    clearCanvas();
    emptyState.hidden = false;
    emptyState.querySelector("strong").textContent = "等待事件视频";
    emptyState.querySelector("span").textContent = "当前筛选范围内暂无可复核报警";
    setText("event-camera", "未选择事件");
    renderEvidenceChain(null, []);
    renderReviewControls(null);
  }

  function resizeCanvas() {
    const width = video.clientWidth;
    const height = video.clientHeight;
    if (width > 0 && height > 0) {
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    drawOverlay();
  }

  function clearCanvas() {
    overlaySlots.forEach(hideOverlaySlot);
  }

  function hideOverlaySlot(slot) {
    slot.box.style.opacity = "0";
    slot.label.style.opacity = "0";
  }

  function ensureOverlaySlot(index) {
    while (overlaySlots.length <= index) {
      const box = document.createElement("div");
      box.className = "event-overlay-box";
      box.style.opacity = "0";
      const label = document.createElement("div");
      label.className = "event-overlay-label";
      label.style.opacity = "0";
      canvas.appendChild(box);
      canvas.appendChild(label);
      overlaySlots.push({ box, label });
    }
    return overlaySlots[index];
  }

  async function toggleFullscreen() {
    if (document.fullscreenElement === videoShell) {
      await document.exitFullscreen();
      return;
    }
    if (videoShell.requestFullscreen) await videoShell.requestFullscreen();
  }

  function toggleVideoPlayback() {
    if (!video.src) return;
    if (video.paused) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }

  function formatPlaybackTime(seconds) {
    const totalSeconds = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
    const minutes = Math.floor(totalSeconds / 60);
    const remainder = totalSeconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
  }

  function syncFullscreenVideoControls() {
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    const currentTime = Math.min(Math.max(0, video.currentTime || 0), duration || 0);
    const playLabel = video.paused ? "▶ 播放" : "Ⅱ 暂停";
    const progressDisabled = duration <= 0;
    const progressValue = duration > 0 ? String(Math.round((currentTime / duration) * 1000)) : "0";
    const timeLabel = `${formatPlaybackTime(currentTime)} / ${formatPlaybackTime(duration)}`;
    if (fullscreenPlayToggle.textContent !== playLabel) fullscreenPlayToggle.textContent = playLabel;
    if (fullscreenProgress.disabled !== progressDisabled) fullscreenProgress.disabled = progressDisabled;
    if (fullscreenProgress.value !== progressValue) fullscreenProgress.value = progressValue;
    if (fullscreenTime.textContent !== timeLabel) fullscreenTime.textContent = timeLabel;
  }

  function updatePlaybackProgressFrame() {
    if (!playbackProgressAnimationActive) return;
    syncFullscreenVideoControls();
    if (video.paused || video.ended) {
      playbackProgressAnimationActive = false;
      return;
    }
    window.requestAnimationFrame(updatePlaybackProgressFrame);
  }

  function startPlaybackProgressAnimation() {
    if (playbackProgressAnimationActive) return;
    playbackProgressAnimationActive = true;
    window.requestAnimationFrame(updatePlaybackProgressFrame);
  }

  function stopPlaybackProgressAnimation() {
    playbackProgressAnimationActive = false;
  }

  function clearFullscreenControlsHideTimer() {
    if (fullscreenControlsHideTimer !== null) {
      window.clearTimeout(fullscreenControlsHideTimer);
      fullscreenControlsHideTimer = null;
    }
  }

  function showFullscreenControls() {
    if (document.fullscreenElement !== videoShell) return;
    videoShell.classList.remove("controls-hidden");
    clearFullscreenControlsHideTimer();
    if (!video.paused) {
      fullscreenControlsHideTimer = window.setTimeout(() => {
        if (document.fullscreenElement === videoShell && !video.paused) {
          videoShell.classList.add("controls-hidden");
        }
        fullscreenControlsHideTimer = null;
      }, 2000);
    }
  }

  function syncFullscreenState() {
    const active = document.fullscreenElement === videoShell;
    // Native WebKit media controls darken the entire video on hover.  Keep
    // them disabled in both embedded and fullscreen modes and use the stable
    // page controls below instead.
    video.controls = false;
    fullscreenVideoControls.hidden = false;
    fullscreenButton.textContent = active ? "⛶" : "⛶";
    fullscreenButton.title = active ? "退出全屏" : "全屏显示";
    fullscreenButton.setAttribute("aria-label", fullscreenButton.title);
    if (active) {
      showFullscreenControls();
    } else {
      clearFullscreenControlsHideTimer();
      videoShell.classList.remove("controls-hidden");
    }
    syncFullscreenVideoControls();
    window.requestAnimationFrame(resizeCanvas);
  }

  function videoRenderRect() {
    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    const stageWidth = canvas.clientWidth;
    const stageHeight = canvas.clientHeight;
    if (!sourceWidth || !sourceHeight || !stageWidth || !stageHeight) return null;
    const scale = Math.min(stageWidth / sourceWidth, stageHeight / sourceHeight);
    const width = sourceWidth * scale;
    const height = sourceHeight * scale;
    return {
      scale,
      width,
      height,
      x: (stageWidth - width) / 2,
      y: (stageHeight - height) / 2,
    };
  }

  function drawOverlay() {
    if (!video.videoWidth || !video.videoHeight || !overlayTimeline.length) {
      clearCanvas();
      return;
    }
    const samples = selectOverlaySamples(
      overlayTimeline,
      video.currentTime,
      MAX_OVERLAY_AGE_SECONDS,
    );
    let visibleCount = 0;
    samples.forEach((sample) => {
      overlayBoxesForSample(sample).forEach((item) => {
        if (drawBox(item, visibleCount)) visibleCount += 1;
      });
    });
    overlaySlots.slice(visibleCount).forEach(hideOverlaySlot);
  }

  function drawBox(item, slotIndex) {
    const box = item.bbox || item.box;
    if (!Array.isArray(box) || box.length < 4) return false;
    const label = String(item.label || item.type || "person").toLowerCase();
    if (label === "person" && !showAlarmBoxes.checked) return false;
    if (label === "phone" && !showPhoneBoxes.checked) return false;
    if (label !== "person" && label !== "phone" && !showContextBoxes.checked) return false;
    const style = boxStyles[label] || boxStyles.person;
    const rect = videoRenderRect();
    if (!rect) return false;
    const [x1, y1, x2, y2] = box.map(Number);
    if (![x1, y1, x2, y2].every(Number.isFinite) || x2 <= x1 || y2 <= y1) return false;
    // Browser-compatible proxies are 1920x1080 while the evidence boxes are
    // recorded in the original camera coordinate space (normally 2560x1440).
    // Use the saved source dimensions whenever present so both versions draw
    // the same overlay at the correct position.
    const sourceWidth = Number(item.frame_width) || video.videoWidth;
    const sourceHeight = Number(item.frame_height) || video.videoHeight;
    if (sourceWidth <= 0 || sourceHeight <= 0) return false;
    const xScale = rect.width / sourceWidth;
    const yScale = rect.height / sourceHeight;
    const x = rect.x + x1 * xScale;
    const y = rect.y + y1 * yScale;
    const width = (x2 - x1) * xScale;
    const height = (y2 - y1) * yScale;
    const lineWidth = Math.max(2, canvas.clientWidth / 720);
    const slot = ensureOverlaySlot(slotIndex);
    const boxElement = slot.box;
    boxElement.style.left = `${x}px`;
    boxElement.style.top = `${y}px`;
    boxElement.style.width = `${width}px`;
    boxElement.style.height = `${height}px`;
    boxElement.style.border = `${lineWidth}px solid ${style.color}`;
    boxElement.style.opacity = "1";
    // Phone detections are deliberately outline-only: the small target stays
    // visible without a label obscuring the surrounding hand or screen.
    const labelElement = slot.label;
    if (label === "phone") {
      labelElement.style.opacity = "0";
      return true;
    }
    labelElement.textContent = style.label;
    labelElement.style.left = `${x}px`;
    labelElement.style.top = `${Math.max(0, y - Math.max(22, canvas.clientHeight / 27))}px`;
    labelElement.style.fontSize = `${Math.max(12, canvas.clientWidth / 85)}px`;
    labelElement.style.background = style.color;
    labelElement.style.opacity = "1";
    return true;
  }

  function changeDashboardState(nextState, preserveSelected) {
    const calendarMonthChanged = nextState.calendarMonth !== dashboardState.calendarMonth;
    dashboardState = nextState;
    userQueryPending = true;
    showRequestState("正在更新筛选结果…", true);
    // Give view selection immediate visual feedback. The stable button nodes
    // also prevent the 0.5-second status/event refresh from swallowing clicks.
    renderViewFilters();
    if (dashboardState.screenCaptureOnly) renderScreenCaptureTimeFilters();
    else renderHourFilters();
    if (calendarMonthChanged) {
      // Month navigation must feel immediate even while the historical event
      // index is refreshing.  Clear stale counts and render the new month now;
      // refreshEvents() fills the counts as soon as the API response arrives.
      calendarCounts = {};
      renderCalendar();
    }
    pendingRefreshPreservesSelection = pendingRefreshPreservesSelection && preserveSelected;
    if (!preserveSelected) {
      selectedMediaKey = "";
      resetReview();
    }
    refreshGate.begin();
    if (eventRequestController) eventRequestController.abort();
    eventPoller.refresh();
  }

  async function refreshStatus() {
    if (document.hidden) return;
    try {
      renderStatus(await fetchJson("/api/status"));
    } catch (error) {
      if (error.name !== "AbortError" && !cameraManagementRestarting) {
        healthState = applyStatusFailure(healthState);
        renderSystemAlerts();
        const failure = statusFailurePresentation(healthState);
        setText("run-state", failure.text);
        document.getElementById("run-state").className = failure.className;
      }
    }
  }

  async function refreshEvents() {
    if (document.hidden) return;
    const preserveSelected = pendingRefreshPreservesSelection;
    pendingRefreshPreservesSelection = true;
    const generation = refreshGate.begin();
    const requestController = new AbortController();
    eventRequestController = requestController;
    if (!eventsLoaded || userQueryPending) showRequestState(eventsLoaded ? "正在更新筛选结果…" : "正在加载事件…", true);
    try {
      const payload = await fetchJson(buildEventsQuery(dashboardState), requestController.signal);
      if (!refreshGate.isCurrent(generation)) return;
      const applied = applyEventsResponse(dashboardState, payload, preserveSelected);
      dashboardState = applied.state;
      reviewOverrides = reconcileReviewOverrides(applied.events, reviewOverrides);
      events = applyReviewOverrides(applied.events, reviewOverrides);
      eventPagination = applied.pagination;
      eventSelection = payload.selection || eventSelection;
      eventDailyTotal = applied.dailyTotal;
      calendarCounts = applied.calendar;
      hourCounts = applied.hours;
      eventSummary = applied.summary;
      summaryOriginalEvents = applied.events;
      vlmReviewSummary = applied.vlmReview;
      renderEvents();
      eventsLoaded = true;
      userQueryPending = false;
      showRequestState(dashboardState.followLatest ? "实时更新 · 数据变化时自动刷新" : "历史事件 · 已更新", false);
    } catch (error) {
      if (error.name === "AbortError" || !refreshGate.isCurrent(generation)) return;
      showRequestState(eventsLoaded ? "暂时无法更新，以下为上次结果" : "事件加载失败，请重试", false, true);
      if (!eventsLoaded) setText("event-empty", "暂时无法读取事件数据，并非没有事件");
    } finally {
      if (eventRequestController === requestController) eventRequestController = null;
    }
  }

  clearFilterButton.addEventListener("click", () => {
    changeDashboardState(selectView(dashboardState, ""), false);
  });
  calendarTrigger.addEventListener("click", () => {
    const willOpen = calendarPopover.hidden;
    calendarPopover.hidden = !willOpen;
    calendarTrigger.setAttribute("aria-expanded", String(willOpen));
    if (willOpen) renderCalendar();
  });
  calendarPrev.addEventListener("click", () => {
    changeDashboardState(
      selectCalendarMonth(dashboardState, shiftMonth(dashboardState.calendarMonth, -1)),
      true,
    );
  });
  calendarNext.addEventListener("click", () => {
    changeDashboardState(
      selectCalendarMonth(dashboardState, shiftMonth(dashboardState.calendarMonth, 1)),
      true,
    );
  });
  showFalsePositives.addEventListener("change", () => {
    changeDashboardState(
      toggleFalsePositiveVisibility(dashboardState, showFalsePositives.checked),
      false,
    );
  });
  excludeVlmFiltered.addEventListener("change", () => {
    changeDashboardState(
      toggleVlmFilteredExclusion(dashboardState, excludeVlmFiltered.checked),
      false,
    );
  });
  showScreenCaptures.addEventListener("click", () => {
    const enabled = !dashboardState.screenCaptureOnly;
    closeCalendar();
    showScreenCaptures.classList.toggle("is-active", enabled);
    showScreenCaptures.setAttribute("aria-pressed", String(enabled));
    changeDashboardState(
      toggleScreenCaptureOnly(dashboardState, enabled),
      false,
    );
  });
  eventPagePrev.addEventListener("click", () => {
    if (eventPagePrev.disabled) return;
    changeDashboardState(selectPage(dashboardState, eventPagination.page - 1, eventSelection), false);
  });
  eventPageNext.addEventListener("click", () => {
    if (eventPageNext.disabled) return;
    changeDashboardState(selectPage(dashboardState, eventPagination.page + 1, eventSelection), false);
  });
  document.addEventListener("click", (event) => {
    if (!eventCalendar.contains(event.target)) closeCalendar();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeCalendar();
  });
  showAlarmBoxes.addEventListener("change", drawOverlay);
  showPhoneBoxes.addEventListener("change", drawOverlay);
  showContextBoxes.addEventListener("change", drawOverlay);
  reviewControls.forEach((button) => {
    button.addEventListener("click", () => submitReview(
      button.dataset.reviewResult,
      null,
      button.dataset.reviewCategory || null,
    ));
  });
  falsePositiveReasonButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const reason = button.dataset.falsePositiveReason;
      falsePositiveDialog.close();
      submitReview("false_positive", reason);
    });
  });
  fixedObjectManagerOpen.addEventListener("click", openFixedObjectManager);
  cameraManagementOpen.addEventListener("click", openCameraManagement);
  cameraManagementFilters.addEventListener("click", (event) => {
    const button = event.target.closest("[data-camera-management-filter]");
    if (!button) return;
    cameraManagementFilter = button.dataset.cameraManagementFilter || "all";
    refreshCameraManagementOverview();
  });
  cameraManagementDialog.addEventListener("close", clearCameraManagementPreviews);
  cameraManagementAdd.addEventListener("click", addCameraManagementCamera);
  cameraManagementSave.addEventListener("click", saveCameraManagementDraft);
  cameraManagementDiscard.addEventListener("click", discardCameraManagementDraft);
  cameraManagementPublish.addEventListener("click", publishCameraManagementDraft);
  screenCalibrationSelect.addEventListener("change", () => {
    if (!screenCalibration) return;
    screenCalibration.selected = Number(screenCalibrationSelect.value) || 0;
    drawScreenCalibration();
  });
  screenCalibrationAdd.addEventListener("click", () => {
    if (!screenCalibration) return;
    const index = screenCalibration.screens.length + 1;
    const offset = Math.min(index - 1, 4) * 24;
    const points = defaultScreenPolygon(screenCalibration.frameSize).map(([x, y]) => [x + offset, y + offset]);
    screenCalibration.screens.push({ screen_id: `screen_${String(index).padStart(2, "0")}`, screen_poly: points });
    screenCalibration.selected = screenCalibration.screens.length - 1;
    renderScreenCalibrationSelect();
    drawScreenCalibration();
  });
  screenCalibrationRemove.addEventListener("click", () => {
    if (!screenCalibration || screenCalibration.screens.length <= 1) return;
    screenCalibration.screens.splice(screenCalibration.selected, 1);
    screenCalibration.selected = Math.max(0, screenCalibration.selected - 1);
    renderScreenCalibrationSelect();
    drawScreenCalibration();
  });
  screenCalibrationReset.addEventListener("click", () => {
    if (!screenCalibration) return;
    screenCalibration.screens = cloneScreenList(screenCalibration.original);
    if (!screenCalibration.screens.length) {
      screenCalibration.screens.push({ screen_id: "screen_01", screen_poly: defaultScreenPolygon(screenCalibration.frameSize) });
    }
    screenCalibration.selected = 0;
    screenCalibrationError.hidden = true;
    renderScreenCalibrationSelect();
    drawScreenCalibration();
  });
  screenCalibrationSave.addEventListener("click", saveScreenCalibrationEdit);
  screenCalibrationCanvas.addEventListener("pointerdown", (event) => {
    const nearest = nearestCalibrationVertex(event);
    if (!nearest) return;
    draggedCalibrationVertex = nearest.index;
    screenCalibrationCanvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  screenCalibrationCanvas.addEventListener("pointermove", (event) => {
    if (draggedCalibrationVertex === null) return;
    const point = calibrationPointFromEvent(event);
    const screen = activeCalibrationScreen();
    if (!point || !screen) return;
    screen.screen_poly[draggedCalibrationVertex] = point.map((value) => Math.round(value * 100) / 100);
    drawScreenCalibration();
  });
  const stopCalibrationDrag = (event) => {
    if (draggedCalibrationVertex === null) return;
    draggedCalibrationVertex = null;
    if (screenCalibrationCanvas.hasPointerCapture(event.pointerId)) screenCalibrationCanvas.releasePointerCapture(event.pointerId);
  };
  screenCalibrationCanvas.addEventListener("pointerup", stopCalibrationDrag);
  screenCalibrationCanvas.addEventListener("pointercancel", stopCalibrationDrag);
  screenCalibrationCanvas.addEventListener("dblclick", (event) => {
    const point = calibrationPointFromEvent(event);
    const screen = activeCalibrationScreen();
    if (!point || !screen) return;
    screen.screen_poly.push(point.map((value) => Math.round(value * 100) / 100));
    drawScreenCalibration();
    event.preventDefault();
  });
  reviewToastUndo.addEventListener("click", undoHiddenFalsePositive);
  downloadButton.addEventListener("click", async (event) => {
    event.stopPropagation();
    if (!downloadEvent || downloadButton.disabled) return;
    const generation = ++downloadGeneration;
    const selected = downloadEvent;
    const isCurrent = () => downloadGeneration === generation && downloadEvent === selected;
    const id = encodeURIComponent(selected.event_id);
    const url = selected.download_url || (selected.is_historical
      ? `/api/runs/${encodeURIComponent(selected.run_id)}/events/${id}/download`
      : `/api/events/${id}/download`);
    downloadButton.disabled = true;
    downloadButton.textContent = "生成中…";
    downloadButton.setAttribute("aria-busy", "true");
    try {
      const destination = await prepareBoxedDownload(url, {
        request: (target, method) => fetchJson(target, undefined, method),
        onState: state => { downloadButton.textContent = state === "queued" ? "排队中…" : "生成中…"; },
        isCurrent, wait: ms => new Promise(resolve => window.setTimeout(resolve, ms)),
      });
      if (!destination || !isCurrent()) return;
      const link = document.createElement("a");
      link.href = destination;
      link.download = `${selected.event_id}-boxed.mp4`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      downloadButton.textContent = "↓ 下载视频";
    } catch (error) {
      if (isCurrent()) {
        downloadButton.textContent = "下载重试";
        downloadButton.title = error.message || "视频生成失败，请重试";
      }
    } finally {
      if (isCurrent()) {
        downloadButton.disabled = false;
        downloadButton.setAttribute("aria-busy", "false");
      }
    }
  });
  fullscreenButton.addEventListener("click", () => toggleFullscreen().catch(() => {}));
  fullscreenPlayToggle.addEventListener("click", toggleVideoPlayback);
  fullscreenProgress.addEventListener("input", () => {
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    if (duration > 0) video.currentTime = (Number(fullscreenProgress.value) / 1000) * duration;
  });
  fullscreenVideoControls.addEventListener("click", (event) => event.stopPropagation());
  videoShell.addEventListener("mousemove", showFullscreenControls);
  videoShell.addEventListener("pointerdown", showFullscreenControls);
  document.addEventListener("fullscreenchange", syncFullscreenState);
  video.addEventListener("loadedmetadata", () => { resizeCanvas(); syncFullscreenVideoControls(); });
  video.addEventListener("timeupdate", () => { drawOverlay(); syncFullscreenVideoControls(); });
  video.addEventListener("durationchange", syncFullscreenVideoControls);
  video.addEventListener("play", () => {
    syncFullscreenVideoControls();
    startPlaybackProgressAnimation();
    showFullscreenControls();
  });
  video.addEventListener("pause", () => {
    stopPlaybackProgressAnimation();
    syncFullscreenVideoControls();
    clearFullscreenControlsHideTimer();
    videoShell.classList.remove("controls-hidden");
  });
  video.addEventListener("ended", stopPlaybackProgressAnimation);
  video.addEventListener("seeking", drawOverlay);
  window.addEventListener("resize", () => { resizeCanvas(); resizeCalibrationCanvas(); });
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(resizeCanvas).observe(video);
  const statusPoller = createPollingLoop(refreshStatus, () => document.hidden ? 15000 : STATUS_POLL_INTERVAL_MS);
  eventPoller = createPollingLoop(refreshEvents, () => document.hidden ? 15000 : pollIntervalForState(dashboardState));
  document.getElementById("event-retry").addEventListener("click", () => eventPoller.refresh());
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) { statusPoller.refresh(); eventPoller.refresh(); }
  });
  statusPoller.refresh();
  eventPoller.refresh();
}
