import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Menu, RefreshCw, Bell, ShieldCheck, User, LogOut, Settings, UserCheck, Search } from 'lucide-react';
import { Button } from '../common/Button';
import { Tooltip } from '../common/Tooltip';
import { StatusBadge } from '../common/StatusBadge';
import { SystemStatusSummary } from '../../types';
import { useAuth } from '../../context/AuthContext';
import './Header.css';

export interface HeaderProps {
  onOpenMobileMenu: () => void;
  systemStatus: SystemStatusSummary | null;
  onRefreshData?: () => void;
  isRefreshing?: boolean;
  onRequestLogout: () => void;
  onOpenSearch: () => void;
  onOpenNotifications: () => void;
  searchTriggerRef?: React.RefObject<HTMLButtonElement>;
  notificationTriggerRef?: React.RefObject<HTMLButtonElement>;
  mobileMenuTriggerRef?: React.RefObject<HTMLButtonElement>;
}

export const Header: React.FC<HeaderProps> = ({
  onOpenMobileMenu,
  systemStatus,
  onRefreshData,
  isRefreshing = false,
  onRequestLogout,
  onOpenSearch,
  onOpenNotifications,
  searchTriggerRef,
  notificationTriggerRef,
  mobileMenuTriggerRef,
}) => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false);

  const overallStatus = systemStatus?.overall_status || null;
  const displayName = user?.name || 'Met Operator';
  const displayRole = user?.role || 'Administrator';

  return (
    <header className="sg-header" aria-label="Application Top Header">
      <div className="sg-header__left">
        <Tooltip content="Toggle Navigation Menu" position="bottom">
          <Button
            ref={mobileMenuTriggerRef}
            variant="ghost"
            size="sm"
            onClick={onOpenMobileMenu}
            ariaLabel="Open mobile navigation menu"
            className="sg-header__mobile-toggle"
            leftIcon={<Menu size={20} />}
          />
        </Tooltip>

        <div className="sg-header__title-group">
          <h1 className="sg-header__title">SkyGuard AI Station Telemetry</h1>
          <span className="sg-header__subtitle">Real-time Meteorological Operational Center</span>
        </div>
      </div>

      <div className="sg-header__right">
        {/* Global Search Quick Trigger */}
        <button
          ref={searchTriggerRef}
          type="button"
          className="sg-header__search-trigger"
          onClick={onOpenSearch}
          aria-label="Open search dialog (Ctrl+K)"
          title="Search telemetry and stations (Ctrl+K)"
        >
          <Search size={15} className="sg-header__search-icon" aria-hidden="true" />
          <span className="sg-header__search-text">Search...</span>
          <kbd className="sg-header__search-kbd">Ctrl K</kbd>
        </button>

        {/* Overall Health Status Indicator */}
        <div className="sg-header__status">
          <span className="sg-header__status-label">Network Health:</span>
          <StatusBadge
            status={overallStatus ?? 'NORMAL'}
            label={overallStatus ? `STATUS: ${overallStatus}` : 'STATUS: …'}
          />
        </div>

        {/* Action Controls */}
        <div className="sg-header__actions">
          {onRefreshData && (
            <Tooltip content="Refresh Telemetry Data" position="bottom">
              <Button
                variant="ghost"
                size="sm"
                onClick={onRefreshData}
                isLoading={isRefreshing}
                ariaLabel="Refresh telemetry data manual action"
                leftIcon={<RefreshCw size={18} />}
              />
            </Tooltip>
          )}

          <Tooltip content="System Alerts & Notifications" position="bottom">
            <Button
              ref={notificationTriggerRef}
              variant="ghost"
              size="sm"
              onClick={onOpenNotifications}
              ariaLabel="View system notifications"
              className="sg-header__action-btn"
              leftIcon={
                <div className="sg-header__bell-wrapper">
                  <Bell size={18} />
                  <span className="sg-header__bell-dot" />
                </div>
              }
            />
          </Tooltip>

          {/* User Profile Dropdown Pill */}
          <div className="sg-header__user-menu-container">
            <button
              type="button"
              className="sg-header__user-pill"
              onClick={() => setIsProfileMenuOpen(!isProfileMenuOpen)}
              aria-expanded={isProfileMenuOpen}
              aria-label={`Operator menu for ${displayName}`}
            >
              <User size={16} />
              <div className="sg-header__user-info">
                <span className="sg-header__user-name">{displayName}</span>
                <span className="sg-header__user-role">{displayRole}</span>
              </div>
              <ShieldCheck size={14} className="sg-header__verified-icon" />
            </button>

            {isProfileMenuOpen && (
              <div className="sg-header__dropdown-menu" role="menu">
                <button
                  type="button"
                  className="sg-header__dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    setIsProfileMenuOpen(false);
                    navigate('/profile');
                  }}
                >
                  <UserCheck size={16} />
                  <span>My Profile</span>
                </button>
                <button
                  type="button"
                  className="sg-header__dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    setIsProfileMenuOpen(false);
                    navigate('/settings');
                  }}
                >
                  <Settings size={16} />
                  <span>System Preferences</span>
                </button>
                <div className="sg-header__dropdown-divider" />
                <button
                  type="button"
                  className="sg-header__dropdown-item sg-header__dropdown-item--danger"
                  role="menuitem"
                  onClick={() => {
                    setIsProfileMenuOpen(false);
                    onRequestLogout();
                  }}
                >
                  <LogOut size={16} />
                  <span>Sign Out</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
