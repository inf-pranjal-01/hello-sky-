import React, { forwardRef } from 'react';
import './Button.css';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  ariaLabel?: string;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  leftIcon,
  rightIcon,
  ariaLabel,
  disabled,
  className = '',
  ...rest
}, ref) => {
  return (
    <button
      ref={ref}
      type="button"
      className={`sg-button sg-button--${variant} sg-button--${size} ${className}`}
      disabled={disabled || isLoading}
      aria-label={ariaLabel}
      aria-busy={isLoading}
      {...rest}
    >
      {isLoading ? (
        <span className="sg-button__spinner" aria-hidden="true">
          <svg className="sg-spinner-icon" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
          </svg>
        </span>
      ) : leftIcon ? (
        <span className="sg-button__icon-left">{leftIcon}</span>
      ) : null}
      
      <span className="sg-button__content">{children}</span>

      {!isLoading && rightIcon ? (
        <span className="sg-button__icon-right">{rightIcon}</span>
      ) : null}
    </button>
  );
});

Button.displayName = 'Button';
