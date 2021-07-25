    public void bad() throws Throwable
    {
        Integer data;

        while (true)
        {
            /* POTENTIAL FLAW: data is null */
            data = null;
            break;
        }

        while (true)
        {
            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.toString());
            break;
        }
    }
