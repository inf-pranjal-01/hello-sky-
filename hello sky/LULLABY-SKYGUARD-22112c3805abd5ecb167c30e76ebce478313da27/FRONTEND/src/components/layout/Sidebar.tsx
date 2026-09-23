/**
 * Sidebar.tsx — Step 3
 * Application navigation sidebar with:
 *  - StationSelector dropdown integration
 *  - routeRegistry-driven nav items
 *  - Desktop collapse/expand
 *  - Mobile drawer with accessible focus trap (useFocusTrap)
 *  - Backdrop click closes mobile drawer
 *  - Escape closes mobile drawer, focus returns to trigger
 *  - Accessible screen-reader announcements on open/close
 */

import React, { useEffect, useRef } from "react";
import { NavLink } from "react-router-dom";
import { ChevronLeft, ChevronRight, Shield } from "lucide-react";
import { Tooltip } from "../common/Tooltip";
import { Button } from "../common/Button";
import { StationSelector } from "./StationSelector";
import { ROUTE_REGISTRY } from "../../config/routeRegistry";
import { useFocusTrap, announceToScreenReader } from "../../hooks/useFocusTrap";
import "./Sidebar.css";

export interface SidebarProps {
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  isMobileOpen: boolean;
  onCloseMobile: () => void;
  /** Ref to mobile menu trigger button for explicit focus restoration */
  mobileTriggerRef?: React.RefObject<HTMLElement>;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isCollapsed,
  onToggleCollapse,
  isMobileOpen,
  onCloseMobile,
  mobileTriggerRef,
}) => {
  // Focus trap only active when mobile drawer is open, with mobileTriggerRef for focus restoration
  const drawerRef = useFocusTrap<HTMLElement>(isMobileOpen, onCloseMobile, {
    triggerRef: mobileTriggerRef,
  });

  const handleBackdropClick = () => onCloseMobile();

  // Screen-reader announcements for mobile navigation drawer
  useEffect(() => {
    if (isMobileOpen) {
      announceToScreenReader("Navigation menu opened");
    }
  }, [isMobileOpen]);

  const wasMobileOpenRef = useRef(false);
  useEffect(() => {
    if (wasMobileOpenRef.current && !isMobileOpen) {
      announceToScreenReader("Navigation menu closed", "polite");
    }
    wasMobileOpenRef.current = isMobileOpen;
  }, [isMobileOpen]);

  return (
    <>
      {/* Mobile Backdrop */}
      <div
        className={`sg-sidebar-backdrop${isMobileOpen ? " sg-sidebar-backdrop--visible" : ""}`}
        onClick={handleBackdropClick}
        aria-hidden="true"
      />

      <aside
        ref={drawerRef}
        id="primary-sidebar"
        role={isMobileOpen ? "dialog" : "navigation"}
        aria-label={isMobileOpen ? "Navigation drawer" : "Primary application navigation"}
        aria-modal={isMobileOpen ? "true" : undefined}
        className={[
          "sg-sidebar",
          isCollapsed ? "sg-sidebar--collapsed" : "",
          isMobileOpen ? "sg-sidebar--mobile-open" : "",
        ]
          .filter(Boolean)
          .join(" ")}
      >
        {/* Brand */}
        <div className="sg-sidebar__brand" aria-label="SkyGuard AI">
          <div className="sg-sidebar__logo-icon" aria-hidden="true">
            <Shield size={24} className="text-accent" />
          </div>
          {!isCollapsed && (
            <div className="sg-sidebar__brand-text">
              <span className="sg-brand-title">SkyGuard AI</span>
              <span className="sg-brand-sub">Meteorological Telemetry</span>
            </div>
          )}
        </div>

        {/* Station Selector (hidden when collapsed on desktop) */}
        {!isCollapsed && (
          <div className="sg-sidebar__station-selector">
            {/*
             * [API: GET /api/stations — INTEGRATED]
             * StationSelector uses StationContext which connects to real backend when VITE_API_MODE=real.
             */}
            <StationSelector />
          </div>
        )}

        {/* Nav */}
        <nav className="sg-sidebar__nav" aria-label="Application sections">
          <ul className="sg-sidebar__menu" role="list">
            {ROUTE_REGISTRY.map((route) => {
              const Icon = route.icon;
              const linkContent = (
                <NavLink
                  to={route.path}
                  end={route.path === "/"}
                  onClick={isMobileOpen ? onCloseMobile : undefined}
                  className={({ isActive }) =>
                    `sg-sidebar__item ${isActive ? "sg-sidebar__item--active" : ""}`
                  }
                  aria-current={undefined}
                >
                  <span className="sg-sidebar__icon" aria-hidden="true">
                    <Icon size={20} />
                  </span>
                  {!isCollapsed && (
                    <span className="sg-sidebar__label">{route.shortLabel}</span>
                  )}
                </NavLink>
              );

              return (
                <li key={route.path}>
                  {isCollapsed ? (
                    <Tooltip content={route.shortLabel} position="right">
                      {linkContent}
                    </Tooltip>
                  ) : (
                    linkContent
                  )}
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Collapse toggle (desktop only) */}
        <div className="sg-sidebar__footer">
          <Tooltip
            content={
              isCollapsed ? "Expand navigation" : "Collapse navigation"
            }
            position="right"
          >
            <Button
              variant="ghost"
              size="sm"
              onClick={onToggleCollapse}
              ariaLabel={
                isCollapsed ? "Expand navigation sidebar" : "Collapse navigation sidebar"
              }
              className="sg-sidebar__toggle-btn"
              leftIcon={
                isCollapsed ? (
                  <ChevronRight size={18} />
                ) : (
                  <ChevronLeft size={18} />
                )
              }
            >
              {!isCollapsed && <span>Collapse</span>}
            </Button>
          </Tooltip>
        </div>
      </aside>
    </>
  );
};
