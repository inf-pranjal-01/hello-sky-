import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import { ChevronRight, Home } from 'lucide-react';
import { getRouteMeta } from '../../config/routeRegistry';
import './Breadcrumbs.css';

export const Breadcrumbs: React.FC = () => {
  const location = useLocation();
  const routeMeta = getRouteMeta(location.pathname);

  return (
    <nav className="sg-breadcrumbs" aria-label="Breadcrumb Navigation">
      <ol className="sg-breadcrumbs__list">
        <li className="sg-breadcrumbs__item">
          <Link to="/dashboard" className="sg-breadcrumbs__link" aria-label="Home Dashboard">
            <Home size={14} aria-hidden="true" />
            <span>SkyGuard</span>
          </Link>
        </li>
        
        <li className="sg-breadcrumbs__separator" aria-hidden="true">
          <ChevronRight size={14} />
        </li>

        <li className="sg-breadcrumbs__item">
          <span className="sg-breadcrumbs__category">{routeMeta.category}</span>
        </li>

        <li className="sg-breadcrumbs__separator" aria-hidden="true">
          <ChevronRight size={14} />
        </li>

        <li className="sg-breadcrumbs__item sg-breadcrumbs__item--active">
          <span aria-current="page" className="sg-breadcrumbs__current">
            {routeMeta.shortLabel}
          </span>
        </li>
      </ol>
    </nav>
  );
};
