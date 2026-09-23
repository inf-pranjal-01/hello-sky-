/**
 * Notification Drawer
 * 
 * [DEMO NOTIFICATION FEED]
 * Slide-in notifications panel with fully accessible focus management.
 * - Focus moves to the close button on open (first focusable element via useFocusTrap).
 * - Tab / Shift+Tab cycle within the drawer (focus trap).
 * - Escape closes the drawer.
 * - Backdrop click closes the drawer.
 * - Focus returns to the trigger button (triggerRef) on close.
 * - Screen-reader announcements on open/close.
 */

import { useCallback, useEffect, useRef } from "react";
import { Bell, X, AlertTriangle, Info, CheckCircle, Zap } from "lucide-react";
import { useFocusTrap, announceToScreenReader } from "../../hooks/useFocusTrap";
import "./NotificationDrawer.css";

// [DEMO FEED] — client-side demo notification feed
interface MockNotification {
  id: string;
  type: "alert" | "warning" | "info" | "success";
  title: string;
  body: string;
  time: string;
  read: boolean;
}

const MOCK_NOTIFICATIONS: MockNotification[] = [
  {
    id: "n-001",
    type: "alert",
    title: "Critical Threshold Exceeded",
    body: "Station ST-MUM-001: Temperature anomaly detected — 47.2°C, breaching critical level.",
    time: "2 min ago",
    read: false,
  },
  {
    id: "n-002",
    type: "warning",
    title: "Sensor Degraded",
    body: "Station ST-DEL-002: Humidity sensor reporting intermittent data. Maintenance recommended.",
    time: "18 min ago",
    read: false,
  },
  {
    id: "n-003",
    type: "info",
    title: "Scheduled Maintenance",
    body: "Station ST-BLR-003 will undergo routine calibration on 10 Sep 2026, 02:00 IST.",
    time: "1 hr ago",
    read: true,
  },
  {
    id: "n-004",
    type: "success",
    title: "Station Restored",
    body: "Station ST-CHE-004 has returned to NORMAL operational status.",
    time: "3 hr ago",
    read: true,
  },
];

function notificationIcon(type: MockNotification["type"]) {
  switch (type) {
    case "alert": return <AlertTriangle size={16} aria-hidden="true" />;
    case "warning": return <Zap size={16} aria-hidden="true" />;
    case "success": return <CheckCircle size={16} aria-hidden="true" />;
    default: return <Info size={16} aria-hidden="true" />;
  }
}

interface NotificationDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  /** Ref to the trigger button so we can restore focus on close */
  triggerRef?: React.RefObject<HTMLElement>;
}

export function NotificationDrawer({ isOpen, onClose, triggerRef }: NotificationDrawerProps) {
  const drawerRef = useFocusTrap<HTMLDivElement>(isOpen, onClose, {
    triggerRef,
  });
  const unread = MOCK_NOTIFICATIONS.filter((n) => !n.read).length;

  // Close on backdrop click
  const handleBackdropClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) onClose();
    },
    [onClose],
  );

  // Prevent body scroll while drawer is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
      announceToScreenReader("Notifications drawer opened");
    } else {
      document.body.style.overflow = "";
    }
    return () => { document.body.style.overflow = ""; };
  }, [isOpen]);

  // Announce close
  const wasOpenRef = useRef(false);
  useEffect(() => {
    if (wasOpenRef.current && !isOpen) {
      announceToScreenReader("Notifications drawer closed", "polite");
    }
    wasOpenRef.current = isOpen;
  }, [isOpen]);

  return (
    <>
      {/* Backdrop */}
      <div
        className={`sg-drawer-backdrop${isOpen ? " sg-drawer-backdrop--visible" : ""}`}
        onClick={handleBackdropClick}
        aria-hidden="true"
      />

      {/* Drawer panel */}
      <div
        ref={drawerRef}
        id="notification-drawer"
        role="dialog"
        aria-modal="true"
        aria-label={`Notifications — ${unread} unread`}
        className={`sg-notification-drawer${isOpen ? " sg-notification-drawer--open" : ""}`}
      >
        {/* Header */}
        <div className="sg-notification-drawer__header">
          <div className="sg-notification-drawer__title">
            <Bell size={18} aria-hidden="true" />
            <h2>Notifications</h2>
            {unread > 0 && (
              <span className="sg-unread-badge" aria-label={`${unread} unread`}>
                {unread}
              </span>
            )}
          </div>
          <button
            className="sg-drawer-close-btn"
            onClick={onClose}
            aria-label="Close notifications"
          >
            <X size={18} aria-hidden="true" />
          </button>
        </div>

        {/* Notice tag */}
        <div className="sg-notification-drawer__body" role="feed" aria-label="Notification feed">
          <p className="sg-backend-notice">
            <span className="sg-notice-tag">DEMO FEED</span>
            {" "}Simulated operational activity alerts.
          </p>

          {MOCK_NOTIFICATIONS.map((n) => (
            <div
              key={n.id}
              className={`sg-notification-item sg-notification-item--${n.type}${n.read ? " sg-notification-item--read" : ""}`}
              role="article"
              aria-label={`${n.type}: ${n.title}`}
            >
              <span className={`sg-notification-icon sg-notification-icon--${n.type}`}>
                {notificationIcon(n.type)}
              </span>
              <div className="sg-notification-content">
                <p className="sg-notification-title">{n.title}</p>
                <p className="sg-notification-body">{n.body}</p>
                <time className="sg-notification-time">{n.time}</time>
              </div>
              {!n.read && <span className="sg-unread-dot" aria-hidden="true" />}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="sg-notification-drawer__footer">
          <button className="sg-mark-all-btn" disabled aria-label="Mark all notifications as read (unavailable in mock mode)">
            Mark all as read
          </button>
        </div>
      </div>
    </>
  );
}

export default NotificationDrawer;
