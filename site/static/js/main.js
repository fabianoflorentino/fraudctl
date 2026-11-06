/* ============================================================
   fraudctl — interações do site
   Mantido pequeno e sem dependências.
   ============================================================ */

(function () {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const topbar = document.getElementById("topbar");
  const nav = document.getElementById("nav");
  const burger = document.getElementById("burger");
  const toTop = document.getElementById("to-top");

  /* Topbar: adiciona contorno depois do primeiro scroll. */
  function syncTopbar() {
    if (!topbar) return;
    topbar.classList.toggle("is-stuck", window.scrollY > 12);
    if (toTop) {
      toTop.classList.toggle("is-visible", window.scrollY > 600);
    }
  }
  window.addEventListener("scroll", syncTopbar, { passive: true });
  syncTopbar();

  /* Menu mobile. */
  if (burger && nav) {
    burger.addEventListener("click", function () {
      const open = nav.classList.toggle("is-open");
      burger.setAttribute("aria-expanded", String(open));
    });
    nav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        nav.classList.remove("is-open");
        burger.setAttribute("aria-expanded", "false");
      });
    });
  }

  /* Switch de idioma: desliza o knob antes de navegar. */
  document.querySelectorAll(".lang-switch").forEach(function (link) {
    link.addEventListener("click", function (event) {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      if (event.button !== 0) return;
      const href = link.getAttribute("href");
      if (!href) return;

      event.preventDefault();

      const target = new URL(href, window.location.href);
      target.searchParams.set("v", Date.now().toString(36));
      const destination = target.pathname + target.search + target.hash;

      if (reduceMotion) {
        window.location.href = destination;
        return;
      }

      link.classList.toggle("is-en");
      link.classList.add("is-switching");
      window.setTimeout(function () {
        window.location.href = destination;
      }, 380);
    });
  });

  /* Reveal sob demanda. */
  const revealNodes = Array.from(document.querySelectorAll(".reveal"));
  if (reduceMotion || !("IntersectionObserver" in window)) {
    revealNodes.forEach(function (node) {
      node.classList.add("is-visible");
    });
  } else {
    const observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
    );
    revealNodes.forEach(function (node) {
      observer.observe(node);
    });
  }

  /* Voltar ao topo. */
  if (toTop) {
    toTop.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
    });
  }
})();
