import { useEffect, useRef } from "react";

const FOCUSABLE_SELECTORS = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(", ");

export interface UseFocusTrapOptions {
  /**
   * Ref to the element that should receive initial focus when the trap activates.
   * If not provided (or its .current is null), the first focusable element inside
   * the container is focused instead.
   */
  initialFocusRef?: React.RefObject<HTMLElement>;
  /**
   * Ref to the trigger element that should receive focus when the overlay closes.
   * If not provided, focus returns to whichever element had focus immediately
   * before the overlay opened.
   */
  triggerRef?: React.RefObject<HTMLElement>;
}

/**
 * useFocusTrap
 *
 * WCAG-compliant keyboard focus management for overlays / drawers / modals.
 * - Moves focus into the container on open (to initialFocusRef or first focusable child).
 * - Tab / Shift+Tab cycle only within the container.
 * - Escape calls onClose.
 * - Returns focus to triggerRef (if specified) or previously-focused element on close.
 *
 * Does NOT create an inaccessible keyboard trap:
 * focus can leave via Escape (which closes the overlay).
 */
export function useFocusTrap<T extends HTMLElement>(
  isActive: boolean,
  onClose: () => void,
  options?: UseFocusTrapOptions,
) {
  const containerRef = useRef<T>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  // Stable reference for onClose to avoid re-running effect on every render
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    if (!isActive) return;

    // Capture the element that was focused before the overlay opened
    previousFocusRef.current = document.activeElement as HTMLElement;

    // Move focus into the container after a short delay (for CSS animations)
    const focusTimer = setTimeout(() => {
      if (!containerRef.current) return;

      // Prefer the explicit initialFocusRef if provided
      if (options?.initialFocusRef?.current) {
        options.initialFocusRef.current.focus();
        return;
      }

      // Otherwise focus the first focusable element
      const focusable =
        containerRef.current.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS);
      if (focusable.length) focusable[0].focus();
    }, 50);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onCloseRef.current();
        return;
      }

      if (e.key !== "Tab" || !containerRef.current) return;

      const focusable = Array.from(
        containerRef.current.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS),
      );
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      clearTimeout(focusTimer);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isActive, options?.initialFocusRef]);

  // Restore focus when overlay closes
  useEffect(() => {
    if (!isActive) {
      // Prefer explicit triggerRef if provided and mounted
      if (options?.triggerRef?.current) {
        options.triggerRef.current.focus();
        previousFocusRef.current = null;
      } else if (previousFocusRef.current) {
        previousFocusRef.current.focus();
        previousFocusRef.current = null;
      }
    }
  }, [isActive, options?.triggerRef]);

  return containerRef;
}

// ---------------------------------------------------------------------------
// Screen-reader live-region announcement utility
// ---------------------------------------------------------------------------

let liveRegionEl: HTMLElement | null = null;

/**
 * Announce a message to screen readers via an assertive live region.
 * Used exclusively for meaningful overlay state changes:
 * - "Search dialog opened"
 * - "Search dialog closed"
 * - "Notifications drawer opened"
 * - "Notifications drawer closed"
 * - "Navigation menu opened"
 * - "Navigation menu closed"
 */
export function announceToScreenReader(message: string, politeness: "polite" | "assertive" = "assertive"): void {
  if (typeof document === "undefined") return;

  // Create the live region lazily if it doesn't exist
  if (!liveRegionEl) {
    liveRegionEl = document.createElement("div");
    liveRegionEl.setAttribute("role", "status");
    liveRegionEl.setAttribute("aria-live", "assertive");
    liveRegionEl.setAttribute("aria-atomic", "true");
    // Visually hidden but accessible to screen readers
    Object.assign(liveRegionEl.style, {
      position: "absolute",
      width: "1px",
      height: "1px",
      padding: "0",
      margin: "-1px",
      overflow: "hidden",
      clip: "rect(0, 0, 0, 0)",
      whiteSpace: "nowrap",
      border: "0",
    });
    document.body.appendChild(liveRegionEl);
  }

  liveRegionEl.setAttribute("aria-live", politeness);

  // Clear and re-set to trigger announcement in screen readers
  liveRegionEl.textContent = "";
  requestAnimationFrame(() => {
    if (liveRegionEl) liveRegionEl.textContent = message;
  });
}
