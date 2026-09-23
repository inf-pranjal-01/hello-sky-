import React, { useState, useEffect, useRef } from 'react';
import { Outlet, useNavigate } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { Footer } from './Footer';
import { GlobalSearchModal } from './GlobalSearchModal';
import { NotificationDrawer } from './NotificationDrawer';
import { ConfirmationDialog } from '../common/ConfirmationDialog';
import { useSensorPolling } from '../../hooks/useSensorPolling';
import { useAuth } from '../../context/AuthContext';
import './AppShell.css';

export const AppShell: React.FC = () => {
  const navigate = useNavigate();
  const { logout } = useAuth();
  const [isCollapsed, setIsCollapsed] = useState<boolean>(false);
  const [isMobileOpen, setIsMobileOpen] = useState<boolean>(false);
  const [isSearchOpen, setIsSearchOpen] = useState<boolean>(false);
  const [isNotificationsOpen, setIsNotificationsOpen] = useState<boolean>(false);
  const [showLogoutConfirm, setShowLogoutConfirm] = useState<boolean>(false);
  const [isLoggingOut, setIsLoggingOut] = useState<boolean>(false);

  // Trigger refs for focus restoration
  const searchTriggerRef = useRef<HTMLButtonElement>(null);
  const notificationTriggerRef = useRef<HTMLButtonElement>(null);
  const mobileMenuTriggerRef = useRef<HTMLButtonElement>(null);

  // Hook providing telemetry data and status summary
  const { systemStatus, refresh, loading } = useSensorPolling(true);

  // Keyboard shortcut for Global Search: Ctrl+K / Cmd+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setIsSearchOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleConfirmLogout = async () => {
    setIsLoggingOut(true);
    try {
      await logout();
      setShowLogoutConfirm(false);
      navigate('/login', { replace: true });
    } finally {
      setIsLoggingOut(false);
    }
  };

  return (
    <div className="sg-shell">
      <Sidebar
        isCollapsed={isCollapsed}
        onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
        isMobileOpen={isMobileOpen}
        onCloseMobile={() => setIsMobileOpen(false)}
        mobileTriggerRef={mobileMenuTriggerRef}
      />

      <div className="sg-shell__main-wrapper">
        <Header
          onOpenMobileMenu={() => setIsMobileOpen(true)}
          systemStatus={systemStatus}
          onRefreshData={refresh}
          isRefreshing={loading}
          onRequestLogout={() => setShowLogoutConfirm(true)}
          onOpenSearch={() => setIsSearchOpen(true)}
          onOpenNotifications={() => setIsNotificationsOpen(true)}
          searchTriggerRef={searchTriggerRef}
          notificationTriggerRef={notificationTriggerRef}
          mobileMenuTriggerRef={mobileMenuTriggerRef}
        />

        <main id="main-content" className="sg-shell__content" tabIndex={-1}>
          <Outlet />
        </main>

        <Footer />
      </div>

      {/* Global Search Modal */}
      <GlobalSearchModal
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        triggerRef={searchTriggerRef}
      />

      {/* Notification Slide-out Drawer */}
      <NotificationDrawer
        isOpen={isNotificationsOpen}
        onClose={() => setIsNotificationsOpen(false)}
        triggerRef={notificationTriggerRef}
      />

      {/* Logout Confirmation Dialog */}
      <ConfirmationDialog
        isOpen={showLogoutConfirm}
        onClose={() => setShowLogoutConfirm(false)}
        onConfirm={handleConfirmLogout}
        title="Confirm Operator Sign Out"
        message="Are you sure you want to terminate your current operator session? You will be returned to the sign-in screen."
        confirmLabel="Sign Out"
        cancelLabel="Cancel"
        isDanger={true}
        isLoading={isLoggingOut}
      />
    </div>
  );
};
