import React, { useEffect, useId } from 'react';
import { X } from 'lucide-react';
import { Button } from './Button';
import { useFocusTrap, announceToScreenReader } from '../../hooks/useFocusTrap';
import './Modal.css';

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  size?: 'sm' | 'md' | 'lg';
}

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  children,
  footer,
  size = 'md',
}) => {
  const titleId = useId();

  // Focus trap handles: Escape key, focus cycling, and focus restoration
  const modalRef = useFocusTrap<HTMLDivElement>(isOpen, onClose);

  // Lock body scroll while modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
      announceToScreenReader(`Dialog opened: ${title}`);
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, title]);

  // Announce close on unmount when transitioning from open → closed
  useEffect(() => {
    return () => {
      if (!isOpen) {
        announceToScreenReader('Dialog closed', 'polite');
      }
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) onClose();
  };

  return (
    <div className="sg-modal-overlay" onClick={handleBackdropClick} role="presentation">
      <div
        ref={modalRef}
        className={`sg-modal sg-modal--${size}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sg-modal__header">
          <h2 id={titleId} className="sg-modal__title">
            {title}
          </h2>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            ariaLabel="Close modal dialog"
            leftIcon={<X size={18} />}
          />
        </div>
        <div className="sg-modal__body">{children}</div>
        {footer && <div className="sg-modal__footer">{footer}</div>}
      </div>
    </div>
  );
};
