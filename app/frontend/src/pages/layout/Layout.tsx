import { Outlet, NavLink, Link } from "react-router-dom";

import styles from "./Layout.module.css";

const Layout = () => {
  return (
    <div className={styles.layout}>
      <header className={styles.header} role={"banner"}>
        <div className={styles.headerContainer}>
          <Link to="/" className={styles.headerTitleContainer}>
            <h3 className={styles.headerTitle}>
              GPT Chatbot und Suche in Dokumenten
            </h3>
          </Link>
          <nav>
            <ul className={styles.headerNavList}>
              <li>
                <NavLink
                  to="/"
                  className={({ isActive }) =>
                    isActive
                      ? styles.headerNavPageLinkActive
                      : styles.headerNavPageLink
                  }
                >
                  Chat
                </NavLink>
              </li>
              <li className={styles.headerNavLeftMargin}>
                <NavLink
                  to="/qa"
                  className={({ isActive }) =>
                    isActive
                      ? styles.headerNavPageLinkActive
                      : styles.headerNavPageLink
                  }
                >
                  Stelle eine Frage
                </NavLink>
              </li>
            </ul>
          </nav>
          <h4 className={styles.headerRightText}>
            Azure OpenAI + Cognitive Search + SCAI
          </h4>
        </div>
      </header>

      <Outlet />
    </div>
  );
};

export default Layout;
