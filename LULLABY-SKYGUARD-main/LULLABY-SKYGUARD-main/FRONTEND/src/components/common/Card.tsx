import React from 'react';
import './Card.css';

export interface CardProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title'> {
  title?: React.ReactNode;
  subtitle?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  className?: string;
  variant?: 'default' | 'glass' | 'subtle';
}

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  action,
  children,
  footer,
  className = '',
  variant = 'default',
  ...rest
}) => {
  return (
    <div className={`sg-card sg-card--${variant} ${className}`} {...rest}>
      {(title || action || subtitle) && (
        <div className="sg-card__header">
          <div>
            {typeof title === 'string' ? <h3 className="sg-card__title">{title}</h3> : title}
            {subtitle && <p className="sg-card__subtitle">{subtitle}</p>}
          </div>
          {action && <div className="sg-card__action">{action}</div>}
        </div>
      )}
      <div className="sg-card__body">{children}</div>
      {footer && <div className="sg-card__footer">{footer}</div>}
    </div>
  );
};
