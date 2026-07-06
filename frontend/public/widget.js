(function () {
  // Config
  var scriptTag = document.currentScript;
  var token = scriptTag.getAttribute('data-token') || 'default';
  var agentId = scriptTag.getAttribute('data-agent-id');
  var host = new URL(scriptTag.src).origin;

  if (!agentId && token === 'default') {
    console.error('Xagent Widget: Missing data-agent-id attribute.');
    return;
  }

  // Visual Configurations
  var buttonSize = scriptTag.getAttribute('data-button-size') || '60px';
  var buttonColor = scriptTag.getAttribute('data-button-color') || '#000';
  var iconColor = scriptTag.getAttribute('data-icon-color') || '#fff';
  var panelBgColor = scriptTag.getAttribute('data-panel-bg-color') || '#fff';

  // Styles
  var style = document.createElement('style');
  style.innerHTML = `
    .xagent-widget-container {
      position: fixed;
      bottom: 20px;
      right: 20px;
      z-index: 999999;
      font-family: system-ui, -apple-system, sans-serif;
    }

    .xagent-widget-fab {
      width: ${buttonSize};
      height: ${buttonSize};
      border-radius: 50%;
      background-color: ${buttonColor};
      color: ${iconColor};
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      transition: transform 0.2s ease, opacity 0.2s ease;
      border: none;
      outline: none;
      padding: 0;
    }

    .xagent-widget-fab:hover {
      transform: scale(1.05);
      opacity: 0.9;
    }

    .xagent-widget-fab svg {
      width: calc(${buttonSize} * 0.53);
      height: calc(${buttonSize} * 0.53);
      fill: currentColor;
    }

    .xagent-widget-panel {
      position: absolute;
      bottom: calc(${buttonSize} + 20px);
      right: 0;
      width: 380px;
      height: 600px;
      max-height: calc(100vh - 100px);
      background: ${panelBgColor};
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
      overflow: hidden;
      opacity: 0;
      visibility: hidden;
      transform: translateY(20px);
      transition: opacity 0.3s ease, transform 0.3s ease, visibility 0.3s;
      border: 1px solid rgba(0,0,0,0.1);
    }

    .xagent-widget-panel.open {
      opacity: 1;
      visibility: visible;
      transform: translateY(0);
    }

    .xagent-widget-iframe {
      width: 100%;
      height: 100%;
      border: none;
      background: transparent;
    }

    @media (max-width: 480px) {
      .xagent-widget-panel {
        width: calc(100vw - 40px);
        height: calc(100vh - 120px);
      }
    }
  `;
  document.head.appendChild(style);

  // Container
  var container = document.createElement('div');
  container.className = 'xagent-widget-container';

  // Panel
  var panel = document.createElement('div');
  panel.className = 'xagent-widget-panel';

  // Generate guest_id if not exists
  var guestId = localStorage.getItem('xagent_guest_id');
  if (!guestId) {
    guestId = 'guest_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
    localStorage.setItem('xagent_guest_id', guestId);
  }

  // Iframe
  var iframe = document.createElement('iframe');
  iframe.className = 'xagent-widget-iframe';
  var iframeUrl = host + '/widget/chat/' + token + '?guest_id=' + guestId;
  if (agentId) {
    iframeUrl += '&agent_id=' + encodeURIComponent(agentId);
  }
  panel.appendChild(iframe);

  function loadIframe(ticket) {
    iframe.src = ticket
      ? iframeUrl + '&embed_ticket=' + encodeURIComponent(ticket)
      : iframeUrl;
  }

  // Request a short-lived embed ticket from the top-level page. This fetch
  // carries the embedding page's real, browser-enforced Origin header, which
  // the backend validates against allowed_domains before signing the ticket.
  // Fetches inside the iframe carry the iframe's own origin instead, so the
  // ticket is how the validated embedding origin reaches the auth call.
  var numericAgentId = parseInt(agentId, 10);
  if (agentId && isNaN(numericAgentId)) {
    console.error('Xagent Widget: data-agent-id must be numeric, got "' + agentId + '".');
    loadIframe(null);
  } else if (agentId) {
    fetch(host + '/api/widget/embed-ticket', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent_id: numericAgentId })
    })
      .then(function (res) {
        if (!res.ok) {
          console.warn('Xagent Widget: embed ticket request failed (HTTP ' + res.status + '); loading widget without a ticket. If this page is not in the agent\'s allowed domains, authentication will fail.');
          return null;
        }
        return res.json();
      })
      .then(function (data) { loadIframe(data && data.ticket); })
      .catch(function (err) {
        console.warn('Xagent Widget: embed ticket request failed (' + err + '); loading widget without a ticket.');
        loadIframe(null);
      });
  } else {
    loadIframe(null);
  }

  // FAB
  var fab = document.createElement('button');
  fab.className = 'xagent-widget-fab';
  // Chat icon SVG
  var chatIcon = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>';
  // Close icon SVG
  var closeIcon = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>';

  fab.innerHTML = chatIcon;

  var isOpen = false;
  fab.onclick = function () {
    isOpen = !isOpen;
    if (isOpen) {
      panel.classList.add('open');
      fab.innerHTML = closeIcon;
    } else {
      panel.classList.remove('open');
      fab.innerHTML = chatIcon;
    }
  };

  container.appendChild(panel);
  container.appendChild(fab);
  document.body.appendChild(container);
})();
