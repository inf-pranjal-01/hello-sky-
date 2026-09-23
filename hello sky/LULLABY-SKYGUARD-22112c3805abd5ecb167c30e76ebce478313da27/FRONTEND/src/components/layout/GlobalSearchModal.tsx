/**
 * Global Search Modal (Ctrl+K / Cmd+K)
 * 
 * [LOCAL NAVIGATION SEARCH INDEX]
 *
 * Full-screen search modal with accessible focus management.
 * - Focus moves to the search input on open (via useFocusTrap initialFocusRef).
 * - Tab / Shift+Tab cycle within the modal (focus trap).
 * - Escape closes the modal.
 * - Backdrop click closes the modal.
 * - Focus returns to the trigger button (triggerRef) or previously focused element on close.
 * - Screen-reader announcements on open/close.
 */

import { useState, useCallback, useRef, useEffect } from "react";
import { Search, X, MapPin, Activity, BarChart2, Wrench, Settings } from "lucide-react";
import { useFocusTrap, announceToScreenReader } from "../../hooks/useFocusTrap";
import { useNavigate } from "react-router-dom";
import "./GlobalSearchModal.css";

// [LOCAL INDEX] — client-side searchable routes and stations interface
interface SearchResult {
  id: string;
  label: string;
  description: string;
  route: string;
  category: string;
  icon: React.ReactNode;
}

const ALL_RESULTS: SearchResult[] = [
  { id: "r-dashboard", label: "Dashboard", description: "Live overview of all station metrics", route: "/", category: "Pages", icon: <Activity size={15} aria-hidden="true" /> },
  { id: "r-monitor", label: "Live Monitor", description: "Real-time sensor data feeds", route: "/monitor", category: "Pages", icon: <Activity size={15} aria-hidden="true" /> },
  { id: "r-stations", label: "Stations", description: "Station inventory and status map", route: "/stations", category: "Pages", icon: <MapPin size={15} aria-hidden="true" /> },
  { id: "r-alerts", label: "Alerts", description: "Triggered alerts and incident log", route: "/alerts", category: "Pages", icon: <Activity size={15} aria-hidden="true" /> },
  { id: "r-analytics", label: "Analytics", description: "Historical trends and data analysis", route: "/analytics", category: "Pages", icon: <BarChart2 size={15} aria-hidden="true" /> },
  { id: "r-reports", label: "Reports", description: "Generated station reports", route: "/reports", category: "Pages", icon: <BarChart2 size={15} aria-hidden="true" /> },
  { id: "r-sensor", label: "Sensor Health", description: "Per-sensor diagnostics and uptime", route: "/sensor-health", category: "Pages", icon: <Activity size={15} aria-hidden="true" /> },
  { id: "r-maintenance", label: "Maintenance", description: "Planned and active maintenance tasks", route: "/maintenance", category: "Pages", icon: <Wrench size={15} aria-hidden="true" /> },
  { id: "r-settings", label: "Settings", description: "Application and alert configuration", route: "/settings", category: "Pages", icon: <Settings size={15} aria-hidden="true" /> },
];

interface GlobalSearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  triggerRef?: React.RefObject<HTMLElement>;
}

export function GlobalSearchModal({ isOpen, onClose, triggerRef }: GlobalSearchModalProps) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const handleClose = useCallback(() => {
    setQuery("");
    onClose();
  }, [onClose]);

  // Focus trap with initial focus directed to the search input, and triggerRef for focus restoration
  const modalRef = useFocusTrap<HTMLDivElement>(isOpen, handleClose, {
    initialFocusRef: inputRef,
    triggerRef,
  });

  // Screen-reader announcements (concise and meaningful)
  useEffect(() => {
    if (isOpen) {
      announceToScreenReader("Search dialog opened");
    }
  }, [isOpen]);

  const wasOpenRef = useRef(false);
  useEffect(() => {
    if (wasOpenRef.current && !isOpen) {
      announceToScreenReader("Search dialog closed", "polite");
    }
    wasOpenRef.current = isOpen;
  }, [isOpen]);

  const filtered = query.trim()
    ? ALL_RESULTS.filter(
        (r) =>
          r.label.toLowerCase().includes(query.toLowerCase()) ||
          r.description.toLowerCase().includes(query.toLowerCase()),
      )
    : ALL_RESULTS;

  const handleSelect = useCallback(
    (route: string) => {
      navigate(route);
      handleClose();
    },
    [navigate, handleClose],
  );

  const handleBackdropClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget) handleClose();
    },
    [handleClose],
  );

  if (!isOpen) return null;

  return (
    <div
      className="sg-search-backdrop"
      onClick={handleBackdropClick}
      role="presentation"
    >
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-label="Global search"
        className="sg-search-modal"
      >
        {/* Search input row */}
        <div className="sg-search-modal__input-row">
          <Search size={18} className="sg-search-modal__icon" aria-hidden="true" />
          <input
            ref={inputRef}
            type="search"
            placeholder="Search pages, stations, sensors…"
            className="sg-search-modal__input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoComplete="off"
            spellCheck={false}
            aria-label="Search query"
          />
          <button
            className="sg-search-modal__close-btn"
            onClick={handleClose}
            aria-label="Close search"
          >
            <X size={16} aria-hidden="true" />
            <kbd className="sg-kbd">Esc</kbd>
          </button>
        </div>

        {/* Results */}
        <div className="sg-search-modal__results" role="listbox" aria-label="Search results">
          <p className="sg-search-notice-tag">
            <span className="sg-notice-tag-inline">LOCAL INDEX</span>
            {" "}Navigation search across active system routes.
          </p>
          {filtered.length === 0 && (
            <p className="sg-search-no-results">No results for &ldquo;{query}&rdquo;</p>
          )}
          {filtered.map((r) => (
            <button
              key={r.id}
              className="sg-search-result-item"
              role="option"
              aria-selected="false"
              onClick={() => handleSelect(r.route)}
            >
              <span className="sg-search-result-icon">{r.icon}</span>
              <span className="sg-search-result-text">
                <span className="sg-search-result-label">{r.label}</span>
                <span className="sg-search-result-desc">{r.description}</span>
              </span>
              <span className="sg-search-result-category">{r.category}</span>
            </button>
          ))}
        </div>

        <div className="sg-search-modal__footer">
          <span className="sg-kbd-hint"><kbd className="sg-kbd">↑</kbd><kbd className="sg-kbd">↓</kbd> navigate</span>
          <span className="sg-kbd-hint"><kbd className="sg-kbd">↵</kbd> open</span>
          <span className="sg-kbd-hint"><kbd className="sg-kbd">Esc</kbd> close</span>
        </div>
      </div>
    </div>
  );
}

export default GlobalSearchModal;
