import React, { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

export interface TooltipProps {
  content: string;
  children: React.ReactElement;
  position?: 'top' | 'bottom' | 'left' | 'right';
}

export const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  position = 'top',
}) => {
  const tooltipId = useId();
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [anchor, setAnchor] = useState<DOMRect | null>(null);

  const show = () => {
    if (wrapperRef.current) setAnchor(wrapperRef.current.getBoundingClientRect());
    setVisible(true);
  };

  useEffect(() => {
    if (!visible) return;
    const refreshPosition = () => {
      if (wrapperRef.current) setAnchor(wrapperRef.current.getBoundingClientRect());
    };
    window.addEventListener('resize', refreshPosition);
    // Capture catches scrolls in nested page/table containers as well.
    window.addEventListener('scroll', refreshPosition, true);
    return () => {
      window.removeEventListener('resize', refreshPosition);
      window.removeEventListener('scroll', refreshPosition, true);
    };
  }, [visible]);

  // Clone child element to inject aria-describedby for screenreaders
  const childWithAria = React.cloneElement(children, {
    'aria-describedby': tooltipId,
  });

  return (
    <div
      ref={wrapperRef}
      className={`tooltip-wrapper tooltip-wrapper--${position}`}
      onMouseEnter={show}
      onMouseLeave={() => setVisible(false)}
      onFocus={show}
      onBlur={() => setVisible(false)}
    >
      {childWithAria}
      {visible && anchor && createPortal(
        <div
          id={tooltipId}
          role="tooltip"
          className={`tooltip-bubble tooltip-bubble--${position}`}
          style={{ left: anchor.left + anchor.width / 2, top: anchor.top + anchor.height / 2 }}
        >
          {content}
        </div>,
        document.body
      )}
    </div>
  );
};
