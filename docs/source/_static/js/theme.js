(function () {
  var KEY = "pns-theme";
  function current() {
    return document.documentElement.getAttribute("data-theme") || "dark";
  }
  function swapLogo(theme) {
    var img = document.querySelector(".wy-side-nav-search img");
    if (img) {
      img.src = img.src.replace(/logo_(dark|light)\.(png|svg)/, "logo_" + theme + ".$2");
    }
  }
  function label(theme) {
    return theme === "light" ? "Dark" : "Light";
  }
  function apply(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try { localStorage.setItem(KEY, theme); } catch (e) {}
    swapLogo(theme);
    var b = document.getElementById("theme-toggle");
    if (b) b.textContent = label(theme);
  }
  function init() {
    if (document.getElementById("theme-toggle")) return; // never create a second button
    var theme = current();
    swapLogo(theme);
    var btn = document.createElement("button");
    btn.id = "theme-toggle";
    btn.type = "button";
    btn.setAttribute("aria-label", "Toggle light and dark theme");
    btn.textContent = label(theme);
    btn.addEventListener("click", function () {
      apply(current() === "light" ? "dark" : "light");
    });
    document.body.appendChild(btn);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
